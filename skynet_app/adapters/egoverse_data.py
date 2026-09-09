"""Honor EgoVerse's declared outlier switch for local, verified joint episodes."""

import numpy as np
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class JointDataset(MultiDataset):
    def __init__(self, *args, reject_outliers=True, **kwargs):
        self.reject_outliers = reject_outliers
        super().__init__(*args, **kwargs)

    def _check_bounds(self, data, dataset, idx, dataset_name):
        if self.reject_outliers:
            return super()._check_bounds(data, dataset, idx, dataset_name)
        # The native reader also returns optional metadata (for example NaN
        # intrinsics when no projection is needed). Validate only model inputs.
        for key in self.zarr_keys.get(data.get("embodiment"), {}).values():
            if key not in data:
                continue
            values = data[key]
            if not hasattr(values, "dtype") or str(values.dtype) in {"object", "str"}:
                continue
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite recorded values in {key}")
        return None
