import sys
import logging
import copy
import torch
import glob
import re
import numpy as np
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random


def RSIAT_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    sum_seed = 0.0
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        sum_seed += _train(args)
    avg_seed = sum_seed / len(seed_list)
    print('Average Seed Accuracy (CNN):', avg_seed)
    logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))


def find_latest_checkpoint(checkpoint_dir):
    """Return the highest-numbered completed task checkpoint, if one exists."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "task_*.pkl"))
    if not checkpoint_files:
        return None

    def task_number(filepath):
        match = re.search(r"task_(\d+)(?:_\d+)?\.pkl$", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    return max(checkpoint_files, key=task_number)

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    output_root = args.get("output_root", "")
    logs_name = os.path.join(
        output_root, "logs", args["model_name"], args["dataset"],
        str(init_cls), str(args["increment"]),
    )
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(
            args["prefix"], args["seed"], args["convnet_type"],
        ),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)
    model.class_order = list(data_manager._class_order)

    checkpoint_dir = os.path.join(
        output_root,
        "ckpt",
        str(args["prefix"]),
        str(args["dataset"]),
        "{}_{}".format(args["init_cls"], args["increment"]),
    )
    if args.get("isolate_runs", True):
        checkpoint_dir = os.path.join(checkpoint_dir, "seed_{}".format(args["seed"]))
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_task = 0
    if args.get("resume", False):
        resume_path = args.get("resume_path") or find_latest_checkpoint(checkpoint_dir)
        if resume_path and os.path.isfile(resume_path):
            completed_task = model.load_checkpoint(resume_path)
            start_task = completed_task + 1
            logging.info("Resuming experiment from task %d", start_task)
        else:
            logging.info("Resume enabled but no checkpoint was found. Starting from task 0.")

    print()
    cnn_curve = getattr(model, "cnn_curve", {"top1": [], "top5": []})
    nme_curve = getattr(model, "nme_curve", {"top1": [], "top5": []})
    if start_task >= data_manager.nb_tasks:
        if cnn_curve["top1"]:
            return sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        logging.info("All tasks are already complete, but no saved accuracy curve was found.")
        return 0.0

    end_task = data_manager.nb_tasks
    max_tasks_per_run = args.get("max_tasks_per_run")
    if max_tasks_per_run is not None:
        max_tasks_per_run = max(1, int(max_tasks_per_run))
        end_task = min(end_task, start_task + max_tasks_per_run)
        logging.info("This run will process tasks [%d, %d).", start_task, end_task)

    for task in range(start_task, end_task):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()
     
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        model.cnn_curve = cnn_curve
        model.nme_curve = nme_curve

        if args.get("save_checkpoints", True):
            checkpoint_path = os.path.join(checkpoint_dir, "task_{}.pkl".format(task))
            model.save_checkpoint(checkpoint_path)
            if args.get("keep_last_checkpoint", True) and task > 0:
                previous_checkpoint = os.path.join(
                    checkpoint_dir, "task_{}.pkl".format(task - 1)
                )
                if os.path.isfile(previous_checkpoint):
                    os.remove(previous_checkpoint)
                    logging.info("Removed superseded checkpoint: %s", previous_checkpoint)


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if end_task < data_manager.nb_tasks:
        logging.info(
            "Stopped after %d task(s); resume the same seed to continue at task %d.",
            end_task - start_task, end_task,
        )
    return sum(cnn_curve["top1"])/len(cnn_curve["top1"])
       
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
