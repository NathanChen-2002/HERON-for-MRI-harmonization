import os
import torch
import torch.distributed as dist
import argparse
import warnings
from trainer import trainer
from datetime import timedelta
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()

# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='',         type=str, help='Absolute directory containing all experiments')
parser.add_argument('--experiment_description', default='test',   type=str, help='experiment name')
parser.add_argument('--run_description',        default='run_description',     type=str, help='run name')

# ========= Select methods ============
parser.add_argument('--harmonizing_model',       default='HERON',          type=str, help='HERON')
parser.add_argument('--ssl_method',              default='HarmonizationTraining',        type=str, help='VICReg, MultiscalePretraining, HarmonizationTraining')
parser.add_argument('--train_mode',              default='ssl',            type=str, help='ssl')
parser.add_argument('--branch2ssl',              default='all',            type=str, help='brain, multiscale, all')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='ssl_dataset',           type=str, help='ssl_dataset')

# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'',           type=str,   help='The path to the parent directory of the datalist JSON file')
parser.add_argument('--device',                 default='cuda:0',            type=str,   help='cpu or cuda')
parser.add_argument('--hparam_overrides',       default='',     type=str, help='Comma-separated overrides, e.g. "learning_rate=5e-4,L_inv_weight=15"')
parser.add_argument('--seed',                  default=0,                   type=int,   help='Random seed')

args = parser.parse_args()


if __name__ == "__main__":
    # [DDP]
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(hours=1)) 
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        args.rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    trainer = trainer(args)
    trainer.train()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()