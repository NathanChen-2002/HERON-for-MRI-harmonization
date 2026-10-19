import torch
import random
import os
import sys
import logging
import numpy as np
from shutil import copy
from datetime import datetime
import ast


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED) 
    torch.cuda.manual_seed_all(SEED) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

def _logger(logger_name, level=logging.DEBUG):

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, ssl_method, harmonizing_model, train_mode, sslbranch,exp_log_dir, seed = None):

    if train_mode == "ssl":
        if sslbranch != "all":
            log_dir = os.path.join(exp_log_dir, train_mode, sslbranch)
        else:
            log_dir = os.path.join(exp_log_dir, train_mode, "SSL")
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {ssl_method}')
    logger.debug(f'Model:  {harmonizing_model}')
    logger.debug("=" * 45)
    logger.debug(f'SEED: {seed}')
    logger.debug("=" * 45)
    return logger, log_dir

def _unwrap_ddp(module):
    return module.module if hasattr(module, "module") else module

def save_checkpoint(log_dir, model, dataset, sslbranch,dataset_configs, hparams):

    if sslbranch == "brain":
        enc = _unwrap_ddp(model[0])
        proj = _unwrap_ddp(model[1])
        save_dict = {
            "dataset": dataset,
            "configs": dataset_configs.__dict__,
            "hparams": dict(hparams),
            "encoder": enc.state_dict(),
            "projector": proj.state_dict()
        }
    elif sslbranch == "multiscale":
        me = _unwrap_ddp(model[0])
        me_dec = _unwrap_ddp(model[1])
        save_dict = {
            "dataset": dataset,
            "configs": dataset_configs.__dict__,
            "hparams": dict(hparams),
            "discriminator": me.state_dict(),
            "discriminator_decoder": me_dec.state_dict() 
        }
    else: 
        me = _unwrap_ddp(model[0])
        be = _unwrap_ddp(model[1])
        dec = _unwrap_ddp(model[2])
        
        save_dict = {
            "dataset": dataset,
            "configs": dataset_configs.__dict__,
            "hparams": dict(hparams),
            "discriminator": me.state_dict(), 
            "be": be.state_dict(),
            "decoder": dec.state_dict()
        }
    # save classification report
    save_path = os.path.join(log_dir, "checkpoint.pt")

    torch.save(save_dict, save_path)


import collections


def to_device(input, device, non_blocking:bool = False):
    if torch.is_tensor(input):
        return input.to(device=device,non_blocking=non_blocking)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device, non_blocking=non_blocking) 
                for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device,non_blocking=non_blocking) 
                for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def parse_hparam_overrides(raw):
    overrides = {}
    if not raw: 
        return overrides
    for item in raw.split(","): 
        item = item.strip() 
        if not item: 
            continue

        if "=" not in item:
            raise ValueError(f"Invalid hparam override: {item}")
        
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        try: 
            parsed = ast.literal_eval(val)
        except Exception: 
            parsed = val
        overrides[key] = parsed
    return overrides




def copy_Files(destination):

    src_dir = os.path.dirname(__file__)
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy(os.path.join(src_dir, "main.py"), os.path.join(destination_dir, f"main.py"))
    copy(os.path.join(src_dir, "models/helpers.py"), os.path.join(destination_dir, f"helpers.py"))
    copy(os.path.join(src_dir, "dataloader/dataloader.py"), os.path.join(destination_dir, f"dataloader.py"))
    copy(os.path.join(src_dir, "dataloader/dataloader_utils.py"), os.path.join(destination_dir, f"dataloader_utils.py"))
    copy(os.path.join(src_dir, "dataloader/__init__.py"), os.path.join(destination_dir, f"__init__.py"))
    copy(os.path.join(src_dir, "models/models.py"), os.path.join(destination_dir, f"models.py"))
    copy(os.path.join(src_dir, "algorithms.py"), os.path.join(destination_dir, f"algorithms.py"))
    copy(os.path.join(src_dir, "configs/data_configs.py"), os.path.join(destination_dir, f"data_configs.py"))
    copy(os.path.join(src_dir, "configs/hparams.py"), os.path.join(destination_dir, f"hparams.py"))
    copy(os.path.join(src_dir, "trainer.py"), os.path.join(destination_dir, f"trainer.py"))
    copy(os.path.join(src_dir, "Prepocess/data_reader.py"), os.path.join(destination_dir, f"data_reader.py"))
    copy(os.path.join(src_dir, "Prepocess/generate_nii_json.py"), os.path.join(destination_dir, f"generate_nii_json.py"))
    copy(os.path.join(src_dir, "Prepocess/split_traveler_4ssl.py"), os.path.join(destination_dir, f"split_traveler_4ssl.py"))
    copy(os.path.join(src_dir, "utils.py"), os.path.join(destination_dir, f"utils.py"))
    copy(os.path.join(src_dir, "BE_ssl.sh"), os.path.join(destination_dir, f"BE_ssl.sh"))
    copy(os.path.join(src_dir, "ME_ssl.sh"), os.path.join(destination_dir, f"ME_ssl.sh"))
    copy(os.path.join(src_dir, "Decoder_ssl.sh"), os.path.join(destination_dir, f"Decoder_ssl.sh"))
 






