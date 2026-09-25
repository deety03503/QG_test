import copy
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.inc_net import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.qgtm import QGTM


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args.get("init_lr", 0.05)
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        # QKD Hyperparameters từ paper CVPR 2026
        self.lambda_kd = args.get("lambda_kd", 1.0)
        self.lambda_s = args.get("lambda_s", 0.05)
        self.temperature = args.get("temperature", 1.0)
        
        # Quantum-Gated Task Modulation Module (QGTM)
        self.qgtm = QGTM(num_qubits=6, num_layers=2, temperature=self.temperature, svd_dim=12)
        self.task_embeddings = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        self._network.to(self._device)
        self.qgtm.to(self._device)

        # Nếu là task tiếp theo, thêm adapter mới và trích xuất task embedding của task vừa xong
        if self._cur_task > 0:
            logging.info(f"Freezing old adapters and adding a new adapter for task {self._cur_task}")
            # Trích xuất task embedding cho task cũ trước khi thêm adapter mới
            prev_task_emb = self.qgtm.extract_task_embedding(self._network, self._cur_task - 1)
            self.task_embeddings.append(prev_task_emb)
            
            # Thêm adapter mới
            self._network.convnet.add_adapter()
            self._network.to(self._device)

        logging.info(f"Learning on classes {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        epochs = self.args.get("epochs", 20)

        # Tập hợp các tham số cần optimize: Adapter mới + QGTM + Classifier FC
        trainable_params = [
            {'params': [p for name, p in self._network.named_parameters() if p.requires_grad], 'lr': self.init_lr, 'weight_decay': self.weight_decay},
            {'params': self.qgtm.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        ]

        optimizer = optim.SGD(trainable_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.min_lr)

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            self.qgtm.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # 1. Feature extraction với Base Adapter A_1 (task_idx=0) cho QGTM (Eq. 2)
                with torch.no_grad():
                    h_t = self._network.convnet.forward_features(inputs, task_idx=0)

                # 2. Forward pass với adapter hiện tại (task_idx=None -> adapter mới nhất)
                logits = self._network(inputs)["logits"]
                loss_ce = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)

                # 3. Nếu cur_task > 0, tính QKD Loss và Sparsity Regularization Loss
                loss_kd = torch.tensor(0.0, device=self._device)
                loss_s = torch.tensor(0.0, device=self._device)

                if self._cur_task > 0 and len(self.task_embeddings) > 0:
                    # Tính alpha_i từ QGTM (Eq. 12)
                    alpha = self.qgtm(h_t, self.task_embeddings) # [B, T_old]
                    
                    if alpha is not None:
                        # Sparsity regularization L_s = ||alpha||_1 (Eq. 11)
                        loss_s = alpha.sum(dim=1).mean()

                        # Task-Interaction Knowledge Distillation Loss (Eq. 13)
                        # Tính logits từ các adapter cũ z_t^(i)
                        old_logits_list = []
                        with torch.no_grad():
                            for t_idx in range(len(self.task_embeddings)):
                                old_feat = self._network.convnet.forward_features(inputs, task_idx=t_idx)
                                old_log = self._network.fc(old_feat)["logits"]
                                old_logits_list.append(old_log)

                        # Weighted KL Distillation
                        for t_idx, old_log in enumerate(old_logits_list):
                            w = alpha[:, t_idx].unsqueeze(1)
                            p_old = F.softmax(old_log, dim=1)
                            p_new = F.log_softmax(logits, dim=1)
                            kl = F.kl_div(p_new, p_old, reduction='none').sum(dim=1, keepdim=True)
                            loss_kd = loss_kd + (w * kl).mean()

                # Tổng Loss (Eq. 15)
                loss = loss_ce + self.lambda_kd * loss_kd + self.lambda_s * loss_s

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss: {losses/len(train_loader):.3f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        self.qgtm.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Trong quá trình inference: nếu có nhiều task, dùng QGTM để tính trọng số adaptive fusion
                if self._cur_task > 0 and len(self.task_embeddings) > 0:
                    h_t = model.convnet.forward_features(inputs, task_idx=0)
                    alpha = self.qgtm(h_t, self.task_embeddings) # [B, T_old]
                    
                    # Adaptive adapter fusion
                    avg_alpha = alpha.mean(dim=0).cpu().numpy()
                    # Mở rộng trọng số cho adapter hiện tại
                    weights = list(avg_alpha) + [1.0]
                    norm_weights = [w / sum(weights) for w in weights]
                    
                    outputs = model.convnet.forward_features(inputs, adapter_weights=norm_weights)
                    logits = model.fc(outputs)["logits"]
                else:
                    logits = model(inputs)["logits"]

                predicts = torch.max(logits, dim=1)[1]
                correct += (predicts == targets).sum().cpu()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
