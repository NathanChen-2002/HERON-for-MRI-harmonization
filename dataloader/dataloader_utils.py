import numpy as np
from monai.transforms import Transform
from collections import OrderedDict 
from monai import transforms


class LoadSubjectStackSliced(Transform):

    def __init__(
        self,
        keys=("img",),
        slice_key="#slice",
        mmap_mode=None,             
        dtype=np.float32,           
        allow_missing_keys=False,   
        cache_size=256,               
    ):
        super().__init__()
        self.keys = tuple(keys)
        self.slice_key = slice_key
        self.mmap_mode = mmap_mode
        self.dtype = dtype
        self.allow_missing_keys = allow_missing_keys
        self.cache_size = int(cache_size) if cache_size else 0
        self._cache = OrderedDict()
 
    def _get_array(self, path):
        if self.cache_size <= 0 or self.mmap_mode is None:
            return np.load(path, mmap_mode=self.mmap_mode)
        arr = self._cache.get(path)
        if arr is not None: 
            self._cache.move_to_end(path) 
            return arr 
        arr = np.load(path, mmap_mode=self.mmap_mode)
        self._cache[path] = arr
        if len(self._cache) > self.cache_size: 
            self._cache.popitem(last=False)
        return arr

    def __call__(self, data):
        d = dict(data)
        if self.slice_key not in d:
            raise KeyError(f"LoadSubjectStackSliced: missing slice_key '{self.slice_key}' in data")
        slc = int(d[self.slice_key])  

        for k in self.keys:
            if k not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"LoadSubjectStackSliced: missing key '{k}' in data")

            path = d[k]
            arr = self._get_array(path)  
            if slc < 0 or slc >= arr.shape[0]:
                raise IndexError(f"LoadSubjectStackSliced: slice {slc} out of range for {path}, shape={arr.shape}")
            if arr.ndim == 3:
                if 0 < slc < arr.shape[0] - 1:
                    out = np.array(arr[slc - 1: slc + 2], copy=True)  
                else:
                    prev = np.zeros_like(arr[0]) if slc == 0 else arr[slc - 1]
                    cur = arr[slc]
                    nxt = np.zeros_like(arr[0]) if slc == arr.shape[0] - 1 else arr[slc + 1]
                    out = np.stack([prev, cur, nxt], axis=0)
            else:
                out = np.array(arr[slc]) 

            if self.dtype is not None and out.dtype != self.dtype:
                out = out.astype(self.dtype, copy=False)

            d[k] = out

        return d


def build_brain_transform(cfg):
    return transforms.Compose([
            LoadSubjectStackSliced(keys=("img", "img_aug"), slice_key="#slice", mmap_mode="r"),
            transforms.EnsureChannelFirstd(keys=["img", "img_aug"], channel_dim=cfg.channel_dim),
            transforms.CenterSpatialCropd(keys=["img", "img_aug"], roi_size=[cfg.size_x, cfg.size_y]),
            transforms.SpatialPadd(keys=["img", "img_aug"], spatial_size=[cfg.size_x, cfg.size_y]),

            transforms.ScaleIntensityRanged(
                keys=["img", "img_aug"],
                a_min=cfg.min_graylevel,
                a_max=cfg.max_graylevel,
                b_min=-1.0,
                b_max=1.0,
                clip=False,
                dtype=np.float32,
            ),

            transforms.SomeOf(
                [
                    transforms.RandAdjustContrastd(
                        keys=["img_aug"], prob=1.0, gamma=(0.4, 1.6), retain_stats=True
                    ),
                    transforms.RandScaleIntensityd(
                        keys=["img_aug"], prob=1.0, factors=(-0.25, 0.25)
                    ),
                    transforms.RandShiftIntensityd(
                        keys=["img_aug"], prob=1.0, offsets=(-0.1, 0.1)
                    ),
                    transforms.RandHistogramShiftd(
                        keys=["img_aug"], prob=1.0, num_control_points=(64, 72)
                    ),
                    transforms.RandBiasFieldd(
                        keys=["img_aug"], prob=1.0, degree=3, coeff_range=(0.0, 0.1)
                    ),
                ],
                num_transforms=(3, 5),
                weights=[0.30, 0.25, 0.25, 0.15, 0.05],
                replace=False,
        ),

        transforms.ThresholdIntensityd(keys=["img_aug"], threshold=1.0, above=False, cval=1.0),
        transforms.ThresholdIntensityd(keys=["img_aug"], threshold=-1.0, above=True, cval=-1.0),
        transforms.ToTensord(keys=["img", "img_aug"]),
    ])  
