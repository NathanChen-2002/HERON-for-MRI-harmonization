from torch.utils.data import Sampler 
from monai import transforms, data
import torch
import json
import os, random
import numpy as np
import collections 
from collections import deque
from monai.data import CacheDataset
from monai.data.utils import set_rnd
import torch.distributed as dist
from dataloader.dataloader_utils import LoadSubjectStackSliced, build_brain_transform

def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def _get_ddp_info():
    if not _ddp_is_on():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info() 
    seed = info.seed 
    np.random.seed(seed % (2**32)) 
    random.seed(seed % (2**32)) 
    set_rnd(info.dataset, seed=seed) 


class SameSliceMultiSiteBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size: int,
        min_sites: int = 5,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        max_unique_sites_per_batch = None, 
        min_samples_per_site_in_batch: int = 1, 
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_sites = int(min_sites)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_unique_sites_per_batch = max_unique_sites_per_batch
        self.min_samples_per_site_in_batch = int(min_samples_per_site_in_batch)
        self.epoch = 0 
        self._batches = []
        self._num_samples = 0

        if self.min_samples_per_site_in_batch < 1:
            raise ValueError(
                f"min_samples_per_site_in_batch({self.min_samples_per_site_in_batch}) must be >= 1"
            )
        if self.batch_size < self.min_sites * self.min_samples_per_site_in_batch:
            raise ValueError(
                f"batch_size({self.batch_size}) < min_sites({self.min_sites}) * "
                f"min_samples_per_site_in_batch({self.min_samples_per_site_in_batch}), Unable to satisfy the constraints"
            )
        if self.max_unique_sites_per_batch is not None:
            if self.max_unique_sites_per_batch < self.min_sites:
                raise ValueError(
                    f"max_unique_sites_per_batch({self.max_unique_sites_per_batch}) < min_sites({self.min_sites})"
                )
            if self.max_unique_sites_per_batch > self.batch_size:
                self.max_unique_sites_per_batch = self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._build_batches()

    def _build_batches(self):
        rng = random.Random(self.seed + self.epoch)
        data_list = self.dataset.data

        slice_to_site_to_indices = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, sample in enumerate(data_list):
            slc = str(sample["#slice"])
            site = sample.get("site_id", None)
            if site is None:
                raise KeyError(f"sample missing site_id key for SameSliceMultiSiteBatchSampler")
            slice_to_site_to_indices[slc][str(site)].append(idx) 
        for site_map in slice_to_site_to_indices.values():
            for lst in site_map.values():
                rng.shuffle(lst)

        batches: list[list[int]] = [] 
        slice_keys = list(slice_to_site_to_indices.keys()) 
        rng.shuffle(slice_keys)

        for slc in slice_keys: 
            site_map = slice_to_site_to_indices[slc]

            while True: 
                active_sites = [
                    s for s, lst in site_map.items()
                    if len(lst) >= self.min_samples_per_site_in_batch
                ]
                if len(active_sites) < self.min_sites:
                    break

                rng.shuffle(active_sites)
                target_unique = min(
                    len(active_sites),
                    self.batch_size // self.min_samples_per_site_in_batch,
                )
                if self.max_unique_sites_per_batch is not None:
                    target_unique = min(target_unique, self.max_unique_sites_per_batch)

                chosen_sites = active_sites[:target_unique] 
                batch = []

                for s in chosen_sites:
                    for _ in range(self.min_samples_per_site_in_batch):
                        batch.append(site_map[s].pop())

                q = deque([s for s in chosen_sites if site_map[s]])
                while len(batch) < self.batch_size and q:
                    s = q.popleft()
                    batch.append(site_map[s].pop())
                    if site_map[s]:
                        q.append(s)

                if len(batch) == self.batch_size:
                    batches.append(batch)
                else:
                    break

        rng.shuffle(batches)

        if self.world_size > 1:
            total_batches = (len(batches) // self.world_size) * self.world_size
            batches = batches[:total_batches]
            batches = batches[self.rank::self.world_size]

        self._batches = batches
        self._num_samples = len(self._batches) * self.batch_size
        # [DDP]
    def __iter__(self):
        if not self._batches:
            self._build_batches()
        for batch in self._batches:
            for idx in batch:
                yield idx

    def __len__(self):
        if not self._batches:
            self._build_batches()
        return self._num_samples



def data_generator(data_path, seed,  dataset_configs, hparams, train_mode,ssl_phase):

    batch_size = hparams["batch_size"]
    if train_mode == "ssl":

        rank, world_size = _get_ddp_info()
        if ssl_phase not in ("brain", "multiscale","all"):
            raise ValueError(f"Unknown ssl_phase: {ssl_phase}") 


        if ssl_phase == "brain":
            brain_train_transform = build_brain_transform(dataset_configs)
        cache_transform = transforms.Compose([
            LoadSubjectStackSliced(keys=("img",), slice_key="#slice", mmap_mode="r"),
            transforms.EnsureChannelFirstd(keys=["img"], channel_dim=dataset_configs.channel_dim),
            transforms.CenterSpatialCropd(keys=["img"], roi_size=[dataset_configs.size_x, dataset_configs.size_y]),
            transforms.SpatialPadd(keys=["img"], spatial_size=[dataset_configs.size_x, dataset_configs.size_y]),
            transforms.ScaleIntensityRanged(
                keys=["img"],
                a_min=dataset_configs.min_graylevel,
                a_max=dataset_configs.max_graylevel,
                b_min=-1.0,
                b_max=1.0,
                clip=False,
                dtype=np.float32,
            ),
            transforms.ToTensord(keys=["img"]),
        ])

        ssl_datalist_path = os.path.join(data_path,"preprocessed_datalist.json")
        with (open(ssl_datalist_path, 'r', encoding='utf8') as dic):
            dataset = json.load(dic)
        def _in_slice_range(sample):
            return dataset_configs.cutlowbond <= int(sample["#slice"]) < dataset_configs.cuthighbond
    
        filtered_dataset = [s for s in dataset if _in_slice_range(s)]
        train_set = [s for s in filtered_dataset if s.get("datasplit", "train") == "train"]
        
        if ssl_phase == "all":
            train_dataset = CacheDataset(train_set,  
                                         transform=cache_transform,
                                         cache_rate=dataset_configs.cache_rate, 
                                         num_workers=30  
                                         )
        elif ssl_phase == "brain":
            train_dataset = CacheDataset(train_set,
                                         transform=brain_train_transform,
                                         cache_rate=dataset_configs.cache_rate,
                                         num_workers=30
                                         )
        elif ssl_phase == "multiscale":
            train_dataset = CacheDataset(train_set,
                                         transform=cache_transform,
                                         cache_rate=dataset_configs.cache_rate,
                                         num_workers=30
                                         )
        min_samples_per_site_in_batch = hparams.get("min_samples_per_site_in_batch", 1)
        ssl_sampler = SameSliceMultiSiteBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            min_sites=hparams["min_sites"],
            seed=seed,
            rank=rank,  # [DDP]
            world_size=world_size,
            max_unique_sites_per_batch=hparams.get("max_unique_sites_per_batch", None),
            min_samples_per_site_in_batch=min_samples_per_site_in_batch
        )
        g_loader = torch.Generator()
        g_loader.manual_seed(seed + rank) 
        
        train_loader = data.DataLoader(train_dataset,
                                        batch_size = batch_size,
                                        num_workers=12, 
                                        pin_memory=True,
                                        persistent_workers=True,
                                        prefetch_factor=6,
                                        sampler = ssl_sampler,
                                        shuffle = False,
                                        drop_last=True,  
                                        worker_init_fn=seed_worker, 
                                        generator=g_loader, 
                                       )
        return train_loader

