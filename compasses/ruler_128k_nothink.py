import os

from mmengine.config import read_base

with read_base():
    from .full_think_16k import models, infer, eval  # noqa: F401
    from opencompass.configs.datasets.ruler.ruler_128k_gen import (
        ruler_datasets as ruler_128k_datasets,
    )

datasets = [*ruler_128k_datasets]

system_prompt = (
    os.environ.build()["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

models[0]["system_prompt"] = system_prompt
