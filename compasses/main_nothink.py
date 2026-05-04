import os

from mmengine.config import read_base

with read_base():
    from .full_think_32k import (
        models,
        infer,
        eval,
        ruler_32k_datasets,
        math_datasets,
        humaneval_datasets,
        LCB_datasets,
        mgsm_datasets,
        mbpp_datasets,
    )

datasets = [
    *ruler_32k_datasets,
    *math_datasets,
    *humaneval_datasets,
    *LCB_datasets,
    *mgsm_datasets,
    *mbpp_datasets,
]
