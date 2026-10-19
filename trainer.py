import torch

import os
import collections
import time
import logging
import matplotlib.pyplot as plt
from models.models import get_network_class
from algorithms import get_algorithm_class
from dataloader.dataloader import data_generator
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint
from utils import parse_hparam_overrides
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class trainer(object):
    def __init__(self, args):
        self.ssl_method = args.ssl_method
        self.train_mode = args.train_mode
        self.harmonizing_model = args.harmonizing_model
        assert args.branch2ssl in ["brain","multiscale","all"]
        self.branch2ssl = args.branch2ssl
        if self.train_mode == "ssl":
            if self.ssl_method == "VICReg":
                if self.branch2ssl != "brain":
                    raise ValueError("VICReg does not support branch2ssl='all'. Use 'brain'.")
            elif self.ssl_method == "HarmonizationTraining": 
                self.branch2ssl = "all"
            elif self.ssl_method == "MultiscalePretraining":
                self.branch2ssl = "multiscale"
            else:
                raise ValueError(f"Unknown ssl_method in ssl mode: {self.ssl_method}")

        self.dataset = args.dataset
        
        self.seed = args.seed

        self.device = torch.device(args.device)


        self.rank = getattr(args, "rank", 0)
        self.world_size = getattr(args, "world_size", 1)
        self.local_rank = getattr(args, "local_rank", 0)
        self.is_main_process = self.rank == 0 
        self.hparam_overrides = getattr(args, "hparam_overrides", "")

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        self.save_dir = f'{args.save_dir}'  


        self.data_path = args.data_path 
        self.create_save_dir() 
        self.dataset_configs, self.hparams_class = self.get_configs() 

        if self.dataset == "ssl_dataset":
            self.hparams = {**self.hparams_class.alg_hparams_ssl_dataset[self.harmonizing_model],
                            **self.hparams_class.other_hparams[self.branch2ssl],
                            **self.hparams_class.train_params}

        self.losses_per_epoch = [] 

    def load_data(self): 
        self.train_dl = data_generator(self.data_path, self.seed, 
                                                     self.dataset_configs, self.hparams, self.train_mode,self.branch2ssl)
        
    def train(self):
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        
        if self.is_main_process: 
            copy_Files(self.exp_log_dir) 

        fix_randomness(self.seed)

        if self.is_main_process:
            self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.ssl_method, self.harmonizing_model,
                                                               self.train_mode, self.branch2ssl, self.exp_log_dir, self.seed)
        else:
            subdir = self.branch2ssl if self.branch2ssl != "all" else "SSL"
            self.scenario_log_dir = os.path.join(
                self.exp_log_dir, self.train_mode, subdir)
            self.logger = logging.getLogger(f"rank{self.rank}_noop")
            self.logger.addHandler(logging.NullHandler())

        if self.hparam_overrides:
            overrides = parse_hparam_overrides(self.hparam_overrides)
            for k, v in overrides.items():
                self.hparams[k] = v
            if self.is_main_process: 
                for k, v in overrides.items():
                    self.logger.debug(f"[HPARAM OVERRIDE] {k}={v}")

        self.load_data()
        backbone_be = get_network_class("BrainEncoder")
        backbone_decoder = get_network_class("HarmonizationDecoder")
        backbone_me = get_network_class("MultiscaleEncoder")
        backbone_me_decoder = get_network_class("AuxiliaryDecoder")

        if self.train_mode == "ssl":
            algorithm_class = get_algorithm_class(self.ssl_method)
            if self.ssl_method == "VICReg":
                    algorithm = algorithm_class(backbone_be, self.dataset_configs, 
                                                self.hparams) 
            elif self.ssl_method == "MultiscalePretraining":
                    algorithm = algorithm_class(backbone_me, backbone_me_decoder,
                                                self.dataset_configs, 
                                                self.hparams)              
                    
            elif self.ssl_method == "HarmonizationTraining":
                be_ckpt_path = self.hparams.get(
                    "be_ckpt_path",
                    os.path.join(self.exp_log_dir, "ssl", "brain", "checkpoint.pt"),
                )

                me_ckpt_path = self.hparams.get(
                    "me_ckpt_path",
                    os.path.join(self.exp_log_dir, "ssl", "multiscale", "checkpoint.pt"),
                )
                backbone_me_ = backbone_me(self.dataset_configs, self.hparams)
                backbone_be_ = backbone_be(self.dataset_configs, self.hparams)
                backbone_decoder_ = backbone_decoder(self.dataset_configs, self.hparams)

                chkpoint_be = torch.load(be_ckpt_path, map_location=self.device, weights_only=True)
                backbone_be_.load_state_dict(self._pick_encoder_state_dict(chkpoint_be, "brain"), strict=True)
                checkpoint_me = torch.load(me_ckpt_path, map_location=self.device, weights_only=True)
                backbone_me_.load_state_dict(self._pick_encoder_state_dict(checkpoint_me, "multiscale"), strict=True)
                for p in backbone_me_.parameters():
                    p.requires_grad = False
                for p in backbone_be_.parameters():
                    p.requires_grad = False
                backbone_me_.eval()
                backbone_be_.eval()

                algorithm = algorithm_class(backbone_me_, backbone_be_, 
                                            backbone_decoder_, self.dataset_configs,
                                            self.hparams)
            else:
                raise ValueError(f"Unknown ssl_method in ssl mode: {self.ssl_method}")

        else:
            raise NotImplementedError("Training mode not found: {}".format(self.train_mode))

        algorithm.to(self.device)

        if dist.is_available() and dist.is_initialized():
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            output_device = self.device.index if self.device.type == "cuda" else None

            if self.train_mode == "ssl" and hasattr(algorithm, "VICReg_algorithm"):
                algorithm.VICReg_algorithm = DDP(
                    algorithm.VICReg_algorithm,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=False,
                )
            elif self.train_mode == "ssl" and hasattr(algorithm, "AE_algorithm"):
                algorithm.AE_algorithm = DDP(
                    algorithm.AE_algorithm,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=False,
                )
                
            elif self.train_mode == "ssl" and hasattr(algorithm, "Harm_algorithm"):
                algorithm.Harm_algorithm = DDP(
                    algorithm.Harm_algorithm,
                    device_ids=device_ids,
                    output_device=output_device,
                    find_unused_parameters=False,
                )

        # training..
        self.logger.debug(f"Hyperparameters:{self.hparams}")
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            epoch_start_time = time.time()
            algorithm.train()

            if hasattr(self.train_dl, "sampler") and hasattr(self.train_dl.sampler, "set_epoch"): 

                self.train_dl.sampler.set_epoch(epoch)

            for step, data_list in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.hparams['num_epochs']}", disable=not self.is_main_process, dynamic_ncols=True)):
                if self.train_mode == 'ssl':
                    current_batch_size = data_list["img"].size()[0] 
                    img_original = to_device(data_list["img"], self.device, non_blocking = True)
                    site_ids = data_list.get("site_id", data_list.get("site", None))

                    if self.branch2ssl == "brain":
                        img_pos = to_device(data_list["img_aug"], self.device, non_blocking=True)
                    elif self.branch2ssl == "multiscale":
                        img_pos = None
                    elif self.branch2ssl == "all":
                        img_pos, site_pos_ids = self._build_cross_site_positive_in_batch(img_original, site_ids)
                    else:
                        raise ValueError(f"Unknown ssl branch in training loop: {self.branch2ssl}")

                    update_kwargs = dict(
                        site_ids=site_ids,
                    )
                    if self.branch2ssl == "all":
                        update_kwargs["site_pos_ids"] = site_pos_ids
                    losses, model = algorithm.update(img_original, img_pos, **update_kwargs)
                else:
                    raise ValueError("Other mode except ssl is not ready yet ")

                for key, val in losses.items():
                    loss_avg_meters[key].update(val, current_batch_size)

            if hasattr(algorithm, "on_epoch_end"):
                algorithm.on_epoch_end()

            if self.is_main_process: 
                # logging
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')

                for key, val in loss_avg_meters.items():
                    self.logger.debug(f'{key}\t: {val.avg:2.4f}')

                if "Total_loss" in loss_avg_meters:
                    self.losses_per_epoch.append(loss_avg_meters["Total_loss"].avg)
                

            epoch_duration = time.time() - epoch_start_time
            if self.is_main_process: 
                self.logger.debug(f"[TIME] Epoch {epoch} took {epoch_duration:.2f} seconds")
                self.logger.debug(f'-------------------------------------')
                if self.ssl_method == "HarmonizationTraining" and epoch % 10 == 0 and epoch < self.hparams["num_epochs"]:
                    save_checkpoint(log_dir=self.scenario_log_dir, model=model, dataset=self.dataset, sslbranch=self.branch2ssl, dataset_configs=self.dataset_configs, hparams=self.hparams)
                    self.logger.debug("[CHECKPOINT] Updated checkpoint.pt")

        if self.is_main_process: 
            self.save_loss_curve(self.scenario_log_dir)

        if self.is_main_process and self.train_mode == "ssl":
            save_checkpoint(log_dir=self.scenario_log_dir, model=model, dataset=self.dataset, sslbranch=self.branch2ssl, dataset_configs=self.dataset_configs, hparams=self.hparams) 


    def _build_cross_site_positive_in_batch(self, img_batch, site_ids):

        site_ids_list = [int(s.item()) if torch.is_tensor(s) else int(s) for s in site_ids]

        site_to_indices = {}
        for idx, site in enumerate(site_ids_list):
            site_to_indices.setdefault(site, []).append(idx)

        site_groups = list(site_to_indices.values())
        partner_idx = torch.empty(img_batch.size(0), device=img_batch.device, dtype=torch.long)
        for group_idx, src_group in enumerate(site_groups):
            dst_group = site_groups[(group_idx + 1) % len(site_groups)]
            src = torch.as_tensor(src_group, device=img_batch.device, dtype=torch.long)
            dst = torch.as_tensor(dst_group, device=img_batch.device, dtype=torch.long)
            take = torch.arange(src.numel(), device=img_batch.device) % dst.numel()
            partner_idx[src] = dst.index_select(0, take)
        site_ids_t = torch.as_tensor(site_ids_list, device=img_batch.device, dtype=torch.long)
        return img_batch.index_select(0, partner_idx), site_ids_t.index_select(0, partner_idx)


    def _normalize_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict
        keys = list(state_dict.keys())
        if all(isinstance(k, str) and k.startswith("module.") for k in keys):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _pick_encoder_state_dict(self, ckpt, branch):

        if branch == "multiscale":
            return self._normalize_state_dict(ckpt["discriminator"])
        if branch == "brain":
            return self._normalize_state_dict(ckpt["encoder"])
        raise ValueError(f"Unknown checkpoint branch: {branch}")


    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.ssl_method)

        return dataset_class(), hparams_class()

    def create_save_dir(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def save_loss_curve(self,save_dir):
        """
        Plots the loss curve for the entire training process.
        """
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(self.losses_per_epoch) + 1), self.losses_per_epoch, label="Train Loss", color='blue')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "loss_curve.png")) 
