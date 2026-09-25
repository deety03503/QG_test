import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QGTM(nn.Module):
    """
    Quantum-Gated Task Modulation (QGTM) Module
    Mã hóa thông tin geometric mutual information giữa sample feature h_t và các đại diện task cũ s_i
    thông qua Parameterized Quantum Circuit (PQC).
    """
    def __init__(self, num_qubits=6, num_layers=2, temperature=1.0, svd_dim=12):
        super(QGTM, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.temperature = temperature
        self.svd_dim = svd_dim

        # Parameterized Quantum Circuit rotation parameters theta: [num_layers, num_qubits]
        self.theta = nn.Parameter(torch.randn(num_layers, num_qubits) * 0.1)

        # Projection layer mapping feature dim (768) to num_qubits angles
        self.proj_sample = nn.Linear(768, num_qubits)
        self.proj_task = nn.Linear(768, num_qubits)

    def extract_task_embedding(self, network, task_idx):
        """
        Gom tham số adapter của task_idx trong tất cả các block transformer,
        áp dụng Truncated SVD để rút gọn thành vector đại diện task s_i.
        """
        weights = []
        for blk in network.convnet.blocks:
            if task_idx < len(blk.adapters):
                adapter = blk.adapters[task_idx]
                weights.append(adapter.down_proj.weight.flatten())
                weights.append(adapter.up_proj.weight.flatten())

        if len(weights) == 0:
            return torch.zeros(self.num_qubits, device=network.convnet.cls_token.device)

        # Stack weights thành matrix S_i: [D, K]
        S_i = torch.stack(weights, dim=1) # [dim, K]
        
        # Truncated SVD
        try:
            U, S, V = torch.svd(S_i)
            r = min(self.svd_dim, S.size(0), S_i.size(1))
            U_r = U[:, :r]
            S_r = S[:r]
            # Vector tổng hợp S_i^(r) * 1
            task_vec = torch.matmul(U_r, S_r) # [dim]
        except Exception:
            task_vec = S_i.mean(dim=1)

        # Collapse the flattened adapter summary back to the ViT feature width.
        # Each adapter weight contains repeated 768-wide feature slices.
        feature_dim = self.proj_task.in_features
        if task_vec.numel() % feature_dim != 0:
            raise ValueError(
                f"Task embedding size {task_vec.numel()} is not divisible by "
                f"the feature dimension {feature_dim}."
            )
        task_vec = task_vec.reshape(-1, feature_dim).mean(dim=0)

        # Normalize & project sang space num_qubits
        norm_task_vec = F.normalize(task_vec, p=2, dim=0)
        task_angle = self.proj_task(norm_task_vec.unsqueeze(0)).squeeze(0)
        return F.normalize(task_angle, p=2, dim=0)

    def simulate_quantum_state(self, angles):
        """
        Mô phỏng trạng thái lượng tử |psi(angles; theta)> trong không gian Hilbert (2^q chiều)
        """
        batch_size = angles.size(0)
        state = torch.ones(batch_size, 1, device=angles.device)
        
        for j in range(self.num_qubits):
            theta_j = angles[:, j] / 2.0
            qubit_state = torch.stack([torch.cos(theta_j), torch.sin(theta_j)], dim=1) # [B, 2]
            state = torch.bmm(state.unsqueeze(2), qubit_state.unsqueeze(1)).view(batch_size, -1)

        for l in range(self.num_layers):
            rot_angles = self.theta[l].unsqueeze(0).expand(batch_size, -1) / 2.0
            rot_state = torch.ones(batch_size, 1, device=angles.device)
            for j in range(self.num_qubits):
                q_rot = torch.stack([torch.cos(rot_angles[:, j]), torch.sin(rot_angles[:, j])], dim=1)
                rot_state = torch.bmm(rot_state.unsqueeze(2), q_rot.unsqueeze(1)).view(batch_size, -1)
            state = F.normalize(state * rot_state, p=2, dim=1)

        return state

    def forward(self, sample_features, task_embeddings):
        """
        sample_features: [B, 768] (h_t = f_ViT(x; A_1))
        task_embeddings: list of T_old task vectors [q]
        Trả về alpha: [B, T_old] (Trọng số tương quan sample-to-task)
        """
        batch_size = sample_features.size(0)
        num_old_tasks = len(task_embeddings)
        
        if num_old_tasks == 0:
            return None

        norm_h = F.normalize(sample_features, p=2, dim=1)
        sample_angles = self.proj_sample(norm_h) # [B, q]

        sample_states = self.simulate_quantum_state(sample_angles)

        fidelities = []
        for i in range(num_old_tasks):
            t_emb = task_embeddings[i].unsqueeze(0).expand(batch_size, -1) # [B, q]
            task_states = self.simulate_quantum_state(t_emb) # [B, 2^q]
            
            dot_prod = (sample_states * task_states).sum(dim=1)
            fidelity = dot_prod ** 2
            fidelities.append(fidelity)

        fidelities = torch.stack(fidelities, dim=1)

        alpha = F.softmax(fidelities / self.temperature, dim=1)
        return alpha
