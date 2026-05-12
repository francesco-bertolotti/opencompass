from mmengine.config import read_base

with read_base():
    from .full_think_32k import models, infer, eval, mmlu_datasets  # noqa: F401

datasets = [*mmlu_datasets]
