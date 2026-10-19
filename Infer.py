import os
import json
import collections
import argparse

import numpy as np
import torch
from tqdm import tqdm

from monai import transforms
from monai import data as monai_data
from monai.data import CacheDataset

from configs.data_configs import get_dataset_class
from models.models import get_network_class
from dataloader.dataloader_utils import LoadSubjectStackSliced


DATASET = "ssl_dataset"
DEVICE = "cuda:0"
CUTLOWBOND = 3
CUTHIGHBOND = 156
TEST_BATCH_SIZE = 32
TEST_NUM_WORKERS = 4


def _site_of(sample):
    return str(sample["site"])


def _subject_of(sample):
    return str(sample["subject"])


def _resolve_weights(checkpoint_path, device):
    main_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    return (
        main_ckpt,
        main_ckpt["discriminator"],
        main_ckpt["be"],
        main_ckpt["decoder"],
    )


def _apply_ckpt_meta(dataset_configs, hparams, ckpt):
    for k, v in ckpt["configs"].items():
        setattr(dataset_configs, k, v)

    hparams.update(ckpt["hparams"])


def _load_ssl_datalist(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def _build_i_site_volume_pool(datalist_path, dataset_configs, center_site):
    dataset = _load_ssl_datalist(datalist_path)

    by_subject = {}

    for sample in dataset:
        if sample["datasplit"] != "train":
            continue

        if _site_of(sample) != str(center_site):
            continue

        sid = int(sample["#slice"])

        if not (dataset_configs.cutlowbond <= sid < dataset_configs.cuthighbond):
            continue

        subj = _subject_of(sample)

        if subj not in by_subject:
            by_subject[subj] = {
                "subject": subj,
                "site": _site_of(sample),
                "img": sample["img"],
            }

    return [
        by_subject[subj]
        for subj in sorted(by_subject.keys())
    ]


def _pick_i_site_volume(pool, subj):
    return next(r for r in pool if str(r["subject"]) == str(subj))


def _build_subject_site_groups(datalist_path, train_datalist_path, dataset_configs, center_site):
    dataset = _load_ssl_datalist(datalist_path)

    i_site_pool = _build_i_site_volume_pool(train_datalist_path, dataset_configs, center_site)

    bag = collections.defaultdict(list)

    for sample in dataset:
        if sample["datasplit"] != "test":
            continue

        site = _site_of(sample)

        sid = int(sample["#slice"])

        if not (dataset_configs.cutlowbond <= sid < dataset_configs.cuthighbond):
            continue

        subj = _subject_of(sample)
        bag[(subj, site)].append(dict(sample))

    groups = []

    for subj, site in sorted(bag.keys()):
        recs = sorted(bag[(subj, site)], key=lambda r: int(r["#slice"]))
        groups.append((subj, site, recs))

    return groups, i_site_pool


def _attach_i_site_volume(recs, i_site_volume):
    out = []

    for r in recs:
        new_r = dict(r)
        new_r["img_site"] = i_site_volume["img"]

        out.append(new_r)

    return out


def _build_group_loader(recs, dataset_configs):
    load_keys = ("img", "img_site")

    cache_transform = transforms.Compose([
        LoadSubjectStackSliced(
            keys=load_keys,
            slice_key="#slice",
            mmap_mode="r",
        ),
        transforms.EnsureChannelFirstd(
            keys=list(load_keys),
            channel_dim=dataset_configs.channel_dim,
        ),
        transforms.CenterSpatialCropd(
            keys=list(load_keys),
            roi_size=[dataset_configs.size_x, dataset_configs.size_y],
        ),
        transforms.SpatialPadd(
            keys=list(load_keys),
            spatial_size=[dataset_configs.size_x, dataset_configs.size_y],
        ),
        transforms.ScaleIntensityRanged(
            keys=list(load_keys),
            a_min=dataset_configs.min_graylevel,
            a_max=dataset_configs.max_graylevel,
            b_min=-1.0,
            b_max=1.0,
            clip=False,
            dtype=np.float32,
        ),
        transforms.ToTensord(keys=list(load_keys)),
    ])

    ds = CacheDataset(
        data=recs,
        transform=cache_transform,
        cache_rate=1.0,
    )

    loader = monai_data.DataLoader(
        ds,
        batch_size=int(TEST_BATCH_SIZE),
        num_workers=int(TEST_NUM_WORKERS),
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
        drop_last=False,
    )

    return loader


@torch.no_grad()
def _forward_group(me, be, decoder, loader, n_slices, size_y, size_x, cutlowbond, device):
    rec_3d = np.empty((n_slices, size_x, size_y), dtype=np.float32)

    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        img_site = batch["img_site"].to(device, non_blocking=True)

        F_ref = me(img_site[:, 1:2, :, :])

        b1 = be(img)

        b2, b3, b4 = decoder.B_downblocks(b1)
        rec = decoder.upsample_path(F_ref, b1, b2, b3, b4).clamp(-1.0, 1.0)

        slice_ids = [int(x) for x in batch["#slice"]]
        bs = img.size(0)

        for k in range(bs):
            sid_local = slice_ids[k] - int(cutlowbond)

            rec_3d[sid_local] = rec[k, 0].detach().cpu().numpy()

    return rec_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--test_datalist_path", required=True)
    parser.add_argument("--train_datalist_path", required=True)
    parser.add_argument("--center_site", required=True)
    parser.add_argument("--ref_subject", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(DEVICE)

    dataset_configs = get_dataset_class(DATASET)()
    hparams = {}

    main_ckpt, me_sd, be_sd, dec_sd = _resolve_weights(args.checkpoint_path, device)
    _apply_ckpt_meta(dataset_configs, hparams, main_ckpt)

    dataset_configs.cutlowbond = int(CUTLOWBOND)
    dataset_configs.cuthighbond = int(CUTHIGHBOND)

    dataset_size_x = int(dataset_configs.size_x)
    dataset_size_y = int(dataset_configs.size_y)
    n_slices = int(dataset_configs.cuthighbond - dataset_configs.cutlowbond)

    me_cls = get_network_class("MultiscaleEncoder")
    be_cls = get_network_class("BrainEncoder")
    dec_cls = get_network_class("HarmonizationDecoder")

    me = me_cls(dataset_configs, hparams).to(device).eval()
    be = be_cls(dataset_configs, hparams).to(device).eval()
    decoder = dec_cls(dataset_configs, hparams).to(device).eval()

    me.load_state_dict(me_sd, strict=True)
    be.load_state_dict(be_sd, strict=True)
    decoder.load_state_dict(dec_sd, strict=True)

    groups, i_site_pool = _build_subject_site_groups(
        args.test_datalist_path, args.train_datalist_path, dataset_configs, args.center_site
    )
    i_site_volume = _pick_i_site_volume(i_site_pool, args.ref_subject)

    for gi, (subj, site, recs) in enumerate(tqdm(groups, desc="forward groups", dynamic_ncols=True)):
        recs_with_site = _attach_i_site_volume(recs, i_site_volume)

        loader = _build_group_loader(
            recs_with_site,
            dataset_configs,
        )

        rec_3d = _forward_group(
            me=me,
            be=be,
            decoder=decoder,
            loader=loader,
            n_slices=n_slices,
            size_y=dataset_size_y,
            size_x=dataset_size_x,
            cutlowbond=dataset_configs.cutlowbond,
            device=device,
        )

        np.savez_compressed(
            os.path.join(args.out_dir, f"{gi:05d}.npz"),
            rec_3D=rec_3d,
            subject=subj,
            site=site,
        )


if __name__ == "__main__":
    main()