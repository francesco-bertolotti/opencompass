from mmengine.config import read_base

with read_base():
    from .full_think_16k import models, infer, eval  # noqa: F401
    from opencompass.configs.datasets.ruler.ruler_8k_gen import (
        ruler_datasets as ruler_8k_datasets,
    )
datasets = [*ruler_8k_datasets]

models[0]["system_prompt"] = "thinking off"
