import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch.nn.functional as F
from models.helpers import PointwiseMLP

from torch.optim.lr_scheduler import LambdaLR

def _split_decay_no_decay_params(module):
    decay_params = []
    no_decay_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):


    def __init__(self, configs, hparams):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.hparams = hparams
    def update(self, *args, **kwargs):
        raise NotImplementedError

def _ddp_is_on() -> bool:

    return dist.is_available() and dist.is_initialized()



class VICReg_base(nn.Module):

    def __init__(self, backbone,configs, hparams):
        super(VICReg_base,self).__init__()

        self.hparams = hparams
        self.dim = hparams['dim'] 
        self.hidden_dim = hparams["proj_hidden_dim"] 


        self.backbone = backbone(configs, hparams)

        self.mc_k = self.hparams.get("mc_k", 72)

        self.proj = PointwiseMLP(
            in_dim=self.dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
        )
    
    def _sample_spatial_k(self, h, idx=None):

        n, c, hh, ww = h.shape
        hw = hh * ww
        k = min(self.mc_k, hw)
        if idx is None:
            if _ddp_is_on():
                if dist.get_rank() == 0:
                    idx = torch.randperm(hw, device=h.device)[:k]
                else:
                    idx = torch.empty(k, device=h.device, dtype=torch.long)
                dist.broadcast(idx, src=0)
            else:
                idx = torch.randperm(hw, device=h.device)[:k]

        h_flat = h.flatten(2)
        h_sel = h_flat.index_select(2, idx)
        return h_sel.permute(2, 0, 1).contiguous(), idx 
    
    def _project_pts(self, pts):

        k, n, c = pts.shape
        pts_2d = pts.reshape(k * n, c)     
        z_2d = self.proj(pts_2d)
        z = z_2d.reshape(k, n, -1)
        return z
    
    def forward(self, I_o, I_pos):


        h_o = self.backbone(I_o) 
        h_pos = self.backbone(I_pos)


        h_o_pts, idx = self._sample_spatial_k(h_o)
        h_pos_pts, _ = self._sample_spatial_k(h_pos, idx=idx)

        z_o_pts = self._project_pts(h_o_pts)       
        z_pos_pts = self._project_pts(h_pos_pts)

        out = {
            "z_o_pts": z_o_pts,
            "z_pos_pts": z_pos_pts,
        }
        return out


class VICReg(Algorithm):

    def __init__(self, backbone, configs, hparams):
        super(VICReg,self).__init__(configs, hparams)

        self.configs = configs
        self.hparams = hparams
        self.VICReg_algorithm = VICReg_base(backbone=backbone,configs=self.configs,
                                            hparams=self.hparams)

        self.optimizer = torch.optim.Adam(
            _split_decay_no_decay_params(self.VICReg_algorithm),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )


        num_epochs = hparams["num_epochs"]

        warmup_epochs = hparams.get("warmup_epochs", 10) 
        assert 0 < warmup_epochs < num_epochs, "warmup_epochs must be in (0, num_epochs)"

        start_factor = self.hparams["scheduler_start_factor"]
        min_factor = self.hparams["scheduler_min_factor"]
        main_epochs = max(1, num_epochs - warmup_epochs)

        def lr_lambda(current_epoch):

            if current_epoch < warmup_epochs:
                return start_factor + (1.0 - start_factor) * (current_epoch / warmup_epochs)
            progress = (current_epoch - warmup_epochs) / main_epochs
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda) 

    def on_epoch_end(self):

        if self.scheduler is not None:
            self.scheduler.step()

    def _normalize_site_ids(self, site_ids, device):

        if site_ids is None:
            return None
        if torch.is_tensor(site_ids):
            return site_ids.to(device=device, dtype=torch.long).view(-1)
        return torch.as_tensor(site_ids, device=device, dtype=torch.long).view(-1)

    def _site_prototypes(self, z, site_ids):

        if site_ids is None:
            raise ValueError("site_ids required for site prototype stats")
        if site_ids.numel() != z.size(1):
            raise ValueError(
                f"site_ids length mismatch: got {site_ids.numel()}, expected {z.size(1)}"
            )
        _, inv = torch.unique(site_ids, sorted=True, return_inverse=True)
        num_sites = int(inv.max().item()) + 1
        protos = []
        for s in range(num_sites):
            mask = inv == s
            protos.append(z[:, mask, :].mean(dim=1))
        return torch.stack(protos, dim=1)

    def _center_by_site_mean(self, z, site_ids):

        if site_ids.numel() != z.size(1): 
            raise ValueError(f"site_ids length mismatch: got {site_ids.numel()}, expected {z.size(1)}") 
        _, inv = torch.unique(site_ids, sorted=True, return_inverse=True) 
        num_sites = int(inv.max().item()) + 1  
        means = [] 
        for s in range(num_sites): 
            mask = inv == s 
            mean_s = z[:, mask, :].mean(dim=1) 
            means.append(mean_s) 
        means = torch.stack(means, dim=1) 
        means_per_sample = means[:, inv, :]  
        return z - means_per_sample  


    def _between_site_mean_loss(self, z, site_ids, eps=1e-4, return_lambda=False):

        protos = self._site_prototypes(z, site_ids)  
        std_between = torch.sqrt(protos.var(dim=1) + eps)  

        std_between_lambda = torch.full_like(std_between, float(self.hparams["L_site_mean_lambda"]))

        loss = torch.mean(F.relu(std_between - std_between_lambda))
        if return_lambda: 
            return loss, std_between_lambda.mean() 
        return loss  

    def _std_loss_per_group_hinge(self, z, group_ids, eps=1e-4):
        z = z.float()
        std_gamma = float(self.hparams.get("std_gamma", 1.0))  

        group_ids_t = self._normalize_site_ids(group_ids, z.device)  
  
        _, inv = torch.unique(group_ids_t, sorted=True, return_inverse=True)  
        num_groups = int(inv.max().item()) + 1  
        loss = torch.zeros((), device=z.device)  
        for g in range(num_groups):  
            mask = inv == g  
            z_g = z[:, mask, :]  
            std_g = torch.sqrt(z_g.var(dim=1, unbiased=False) + eps)  
            loss_g = torch.mean(F.relu(std_gamma - std_g))  
            loss = loss + loss_g  
        return loss / float(num_groups)  

    def _cov_loss_gram(self, z):
        z = z.float()
        K, N, D = z.shape
        denom = (N - 1)
        G = torch.bmm(z, z.transpose(1, 2))

        frob_C_sq = (G ** 2).sum(dim=(1, 2)) / (denom ** 2)  


        diagC = (z ** 2).sum(dim=1) / denom             
        diagC_sq_sum = (diagC ** 2).sum(dim=1)       

        offdiag_sq_sum = frob_C_sq - diagC_sq_sum            
        return (offdiag_sq_sum / D).mean()                   
    
    def _prepare_stats(self, x1, x2, site_ids_t):
        x1_stats = self._center_by_site_mean(x1, site_ids_t)
        x2_stats = self._center_by_site_mean(x2, site_ids_t)

        x1_stats = x1_stats - x1_stats.mean(dim=1, keepdim=True)
        x2_stats = x2_stats - x2_stats.mean(dim=1, keepdim=True)
        return x1_stats, x2_stats

    def _std_branch_loss(self, x1_stats, x2_stats, site_ids_t):
        std1 = self._std_loss_per_group_hinge(x1_stats, site_ids_t)
        std2 = self._std_loss_per_group_hinge(x2_stats, site_ids_t)
        return std1 / 2.0 + std2 / 2.0
    
    def _site_mean_branch_loss(self, x1, x2,site_ids_t_site_mean):
        between1, lambda1 = self._between_site_mean_loss(x1, site_ids_t_site_mean, return_lambda=True)
        between2, lambda2 = self._between_site_mean_loss(x2, site_ids_t_site_mean, return_lambda=True)
        branch_loss = between1 / 2.0 + between2 / 2.0
        branch_lambda = lambda1 / 2.0 + lambda2 / 2.0
        return branch_loss, branch_lambda
    
    def update(self, I_o, I_pos, site_ids=None):

        self.optimizer.zero_grad()

        output = self.VICReg_algorithm(I_o,I_pos)

        site_mean_weight = float(self.hparams.get("L_site_mean_weight", 0.0))

        z1 = output["z_o_pts"]
        z2 = output["z_pos_pts"]

        site_ids_t = self._normalize_site_ids(site_ids, z1.device)

        repr_loss = F.mse_loss(z1, z2)

        z1_stats, z2_stats = self._prepare_stats(z1, z2,site_ids_t)

        std_loss = self._std_branch_loss(z1_stats, z2_stats,site_ids_t)

        cov1 = self._cov_loss_gram(z1_stats)
        cov2 = self._cov_loss_gram(z2_stats)
        cov_loss = cov1 + cov2      

        site_mean_loss, site_mean_lambda_log = self._site_mean_branch_loss(z1, z2,site_ids_t)

        # -------- total --------
        loss = (
            self.hparams["L_inv_weight"] * repr_loss
            + self.hparams["L_var_weight"] * std_loss
            + self.hparams["L_cov_weight"] * cov_loss
            + site_mean_weight * site_mean_loss  
        )

        loss.backward()
        self.optimizer.step()

        metric_logs = torch.stack([  
            loss.detach(),  
            repr_loss.detach(),  
            std_loss.detach(),  
            site_mean_loss.detach(),
            cov_loss.detach(),  
            site_mean_lambda_log.detach(),  
        ])  

        if _ddp_is_on(): 
            dist.all_reduce(metric_logs, op=dist.ReduceOp.SUM)  
            metric_logs /= dist.get_world_size()  

        loss_log, repr_loss_log, std_log, site_mean_log, cov_log, site_mean_lambda_avg = metric_logs.unbind(0) 


        if hasattr(self.VICReg_algorithm, "module"):
            backbone = self.VICReg_algorithm.module.backbone
            projector = self.VICReg_algorithm.module.proj
        else:
            backbone = self.VICReg_algorithm.backbone
            projector = self.VICReg_algorithm.proj

        return {
            'Total_loss': loss_log.item(),
            "repr": repr_loss_log.item(),
            "std": std_log.item(),  
            "site_mean": site_mean_log.item(),
            "cov": cov_log.item(),  
            "site_mean_lambda": site_mean_lambda_avg.item(),  
        }, [backbone, projector]
        



class MultiscaleAutoencoder(nn.Module):
    def __init__(self, backbone_me, backbone_me_decoder, configs, hparams):
        super(MultiscaleAutoencoder, self).__init__()

        self.backbone_me = backbone_me(configs, hparams)
        self.backbone_me_decoder =  backbone_me_decoder(configs, hparams)
        self.configs = configs
        self.hparams = hparams

    def forward(self,I):

        F4 = self.backbone_me(I)["F4"]
        rec = self.backbone_me_decoder(F4)
        return rec

class MultiscalePretraining(nn.Module):
    def __init__(self, backbone_me, backbone_me_decoder, configs, hparams):
        super(MultiscalePretraining, self).__init__()

        self.configs = configs
        self.hparams = hparams
        self.AE_algorithm = MultiscaleAutoencoder(backbone_me, backbone_me_decoder, configs, hparams)


        self.optimizer = torch.optim.Adam(
            _split_decay_no_decay_params(self.AE_algorithm),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        num_epochs = hparams["num_epochs"]
        warmup_epochs = hparams.get("warmup_epochs", 10) 
        assert 0 < warmup_epochs < num_epochs, "warmup_epochs must be in (0, num_epochs)"

        start_factor = self.hparams["scheduler_start_factor"]
        min_factor = self.hparams["scheduler_min_factor"]
        main_epochs = max(1, num_epochs - warmup_epochs)

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return start_factor + (1.0 - start_factor) * (current_epoch / warmup_epochs)
            progress = (current_epoch - warmup_epochs) / main_epochs
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def on_epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def update(self, I_o, I_pos=None, site_ids=None):


        self.optimizer.zero_grad()

        I_o_center = I_o[:, 1:2, :, :]
        rec = self.AE_algorithm(I_o_center)

        fg_mask = (I_o_center.detach() > -0.95).float()
        weight = fg_mask + (1.0 - fg_mask) * 0.01
        pixel_l1 = (rec - I_o_center).abs()
        loss = (pixel_l1 * weight).mean()
        
        loss.backward()
        self.optimizer.step()

        metric_logs = torch.stack([
            loss.detach(),
        ])

        if _ddp_is_on():
            dist.all_reduce(metric_logs, op=dist.ReduceOp.SUM)
            metric_logs /= dist.get_world_size()

        loss_log = metric_logs[0]
        if hasattr(self.AE_algorithm, "module"):
            backbone_me = self.AE_algorithm.module.backbone_me
            backbone_me_decoder = self.AE_algorithm.module.backbone_me_decoder
        else:
            backbone_me = self.AE_algorithm.backbone_me
            backbone_me_decoder = self.AE_algorithm.backbone_me_decoder
        return {
            'Total_loss': loss_log.item(),
        }, [backbone_me, backbone_me_decoder]









class HarmonizationModel(nn.Module):
    def __init__(self, backbone_me, backbone_be,
                 backbone_decoder, configs, hparams):
        super(HarmonizationModel, self).__init__()

        self.backbone_me = backbone_me
        self.backbone_be = backbone_be
        self.backbone_decoder = backbone_decoder
        self.configs = configs
        self.hparams = hparams

    def forward(
        self,
        I_src,
        I_ref,
    ):
        with torch.no_grad():
            B_out1 = self.backbone_be(I_src)
            I_src_center = I_src[:, 1:2, :, :]
            I_ref_center = I_ref[:, 1:2, :, :]
            F_src = self.backbone_me(I_src_center)
            F_ref = self.backbone_me(I_ref_center)

        B_out2, B_out3, B_out4 = self.backbone_decoder.B_downblocks(B_out1)
        rec_swap = self.backbone_decoder.upsample_path(F_ref, B_out1, B_out2, B_out3, B_out4)
        rec_id = self.backbone_decoder.upsample_path(F_src, B_out1, B_out2, B_out3, B_out4)


        return {
            "rec_swap": rec_swap,
            "rec_id": rec_id,
            "F_src": F_src,
            "F_ref": F_ref,
            "B_out1": B_out1,
        }
 

class HarmonizationTraining(Algorithm):
    def __init__(self, backbone_me, backbone_be,
                 backbone_decoder, configs, hparams):
        super(HarmonizationTraining, self).__init__(configs, hparams)

        self.configs = configs
        self.hparams = hparams
        self.Harm_algorithm = HarmonizationModel(backbone_me, backbone_be,
                                        backbone_decoder, configs, hparams)

        self.maeloss = nn.L1Loss()

        self.mseloss = nn.MSELoss()

        self.w_ir = float(self.hparams["L_ir_weight"])
        self.w_rs = float(self.hparams["L_rs_weight"])
        self.w_id = float(self.hparams["L_id_weight"])
        self.w_fr = float(self.hparams["L_fr_weight"])
        self.w_site_prototype = float(self.hparams["L_site_prototype_weight"])

        trainable_params = [p for p in self.Harm_algorithm.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("HarmonizationTraining has no trainable parameters. Check freeze settings.")


        self.optimizer = torch.optim.Adam(
            _split_decay_no_decay_params(
                self.Harm_algorithm.backbone_decoder
            ),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )


        num_epochs = hparams["num_epochs"]

        warmup_epochs = hparams.get("warmup_epochs", 10) 
        assert 0 < warmup_epochs < num_epochs, "warmup_epochs must be in (0, num_epochs)"

        start_factor = self.hparams["scheduler_start_factor"]
        min_factor = self.hparams["scheduler_min_factor"]
        main_epochs = max(1, num_epochs - warmup_epochs)

        def lr_lambda(current_epoch):

            if current_epoch < warmup_epochs:
                return start_factor + (1.0 - start_factor) * (current_epoch / warmup_epochs)
            progress = (current_epoch - warmup_epochs) / main_epochs
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda) 

    def on_epoch_end(self):

        if self.scheduler is not None:
            self.scheduler.step()

    def _mu_sigma(self, feat_bchw, eps=1e-5):

        mu = feat_bchw.mean(dim=(2, 3))
        sigma = feat_bchw.var(dim=(2, 3), unbiased=False).clamp_min(eps).sqrt()
        return mu, sigma

    def _multiscale_mu_sigma_loss(self, pred_feats, target_feats):

        total = 0.0
        for key in ("F1", "F2", "F3"):
            mu_pred, sig_pred = self._mu_sigma(pred_feats[key])
            mu_tgt, sig_tgt = self._mu_sigma(target_feats[key])
            total = total + (mu_pred - mu_tgt).pow(2).mean() + (sig_pred - sig_tgt).pow(2).mean()
        return total / 3.0

    def _multiscale_feature_l2(self, pred_feats, target_feats):
        total = 0.0
        for key in ("F1", "F2", "F3"):
            total = total + self.mseloss(pred_feats[key], target_feats[key])
        return total / 3.0

    def _norm_me_feats(self, feats, keys):
        return {key: F.instance_norm(feats[key], weight=None, bias=None) for key in keys}

    def _site_prototypes(self, feats, inv, num_groups, keys):

        return {
            key: torch.stack([
                feats[key][inv == group_idx].mean(dim=0)
                for group_idx in range(num_groups)
            ])
            for key in keys
        }

    def _site_centered_feature_l2(self, pred_feats, target_feats, site_ids_t, keys):
        _, inv = torch.unique(site_ids_t, sorted=True, return_inverse=True)
        num_groups = int(inv.max().item()) + 1
        pred_proto = self._site_prototypes(pred_feats, inv, num_groups, keys)
        target_proto = self._site_prototypes(target_feats, inv, num_groups, keys)
        total = 0.0
        for key in keys:
            total = total + self.mseloss(
                pred_feats[key] - pred_proto[key].index_select(0, inv),
                target_feats[key] - target_proto[key].index_select(0, inv),
            )
        return total / float(len(keys))

    def _site_prototype_loss(self, pred_feats, target_feats, site_ids_t, site_pos_ids_t, keys):

        _, src_inv = torch.unique(site_ids_t, sorted=True, return_inverse=True)
        _, dst_inv = torch.unique(site_pos_ids_t, sorted=True, return_inverse=True)
        num_src = int(src_inv.max().item()) + 1
        num_dst = int(dst_inv.max().item()) + 1
        dst_for_src = torch.stack([
            dst_inv[src_inv == src_idx][0]
            for src_idx in range(num_src)
        ])
        pred_proto = self._site_prototypes(pred_feats, src_inv, num_src, keys)
        target_proto = self._site_prototypes(target_feats, dst_inv, num_dst, keys)
        total = 0.0
        for key in keys:
            total = total + self.mseloss(
                pred_proto[key],
                target_proto[key].index_select(0, dst_for_src),
            )
        return total / float(len(keys))

    def update(self, I_o, I_pos=None, site_ids=None, site_pos_ids=None):

        harm_mod = self.Harm_algorithm.module if hasattr(self.Harm_algorithm, "module") else self.Harm_algorithm
        harm_mod.backbone_me.eval()
        harm_mod.backbone_be.eval()

        
        self.optimizer.zero_grad()
        site_ids_t = site_ids.to(device=I_o.device, dtype=torch.long).view(-1)
        site_pos_ids_t = site_pos_ids.to(device=I_o.device, dtype=torch.long).view(-1)

        I_o_center = I_o[:, 1:2, :, :]
        output = self.Harm_algorithm(I_o, I_pos)
        rec_swap = output["rec_swap"]
        rec_id = output["rec_id"]

        F_swap = harm_mod.backbone_me(rec_swap)
        F_id = harm_mod.backbone_me(rec_id)

        me_loss_keys = ("F1", "F2", "F3")
        F_swap_norm = self._norm_me_feats(F_swap, me_loss_keys)
        F_ref_norm = self._norm_me_feats(output["F_ref"], me_loss_keys)
        F_src_norm = self._norm_me_feats(output["F_src"], me_loss_keys)

        ir_loss = self._site_centered_feature_l2(F_swap_norm, F_src_norm, site_ids_t, me_loss_keys)
        rs_loss = self._multiscale_mu_sigma_loss(F_swap, output["F_ref"])
        id_loss = self.maeloss(rec_id, I_o_center)
        fr_loss = self._multiscale_feature_l2(F_id, output["F_src"])
        site_prototype_loss = self._site_prototype_loss(F_swap_norm, F_ref_norm, site_ids_t, site_pos_ids_t, me_loss_keys)
        loss = (
            self.w_ir * ir_loss
            + self.w_rs * rs_loss
            + self.w_id * id_loss
            + self.w_fr * fr_loss
            + self.w_site_prototype * site_prototype_loss
        )


        loss.backward()
        self.optimizer.step()

        metric_logs = torch.stack([
            loss.detach(),
            ir_loss.detach(),
            rs_loss.detach(),
            id_loss.detach(),
            fr_loss.detach(),
            site_prototype_loss.detach(),
        ])

        if _ddp_is_on():
            dist.all_reduce(metric_logs, op=dist.ReduceOp.SUM)
            metric_logs /= dist.get_world_size()
        
        loss_log, ir_log, rs_log, id_log, fr_log, site_proto_log = metric_logs.unbind(0)

        if hasattr(self.Harm_algorithm, "module"):
            backbone_me = self.Harm_algorithm.module.backbone_me
            backbone_be = self.Harm_algorithm.module.backbone_be
            backbone_decoder = self.Harm_algorithm.module.backbone_decoder
        else:
            backbone_me = self.Harm_algorithm.backbone_me
            backbone_be = self.Harm_algorithm.backbone_be
            backbone_decoder = self.Harm_algorithm.backbone_decoder
        return {
            'Total_loss': loss_log.item(),
            'individual_residual': ir_log.item(),
            'reference_statistics': rs_log.item(),
            'id': id_log.item(),
            'fr': fr_log.item(),
            'site_prototype': site_proto_log.item(),
        }, [backbone_me, backbone_be, backbone_decoder]

