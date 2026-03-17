import os

from mmengine.config import read_base


with read_base():
    from .full_think_16k import models, infer, eval  # noqa: F401
    from opencompass.configs.datasets.ruler.ruler_4k_gen import (
        ruler_datasets as ruler_4k_datasets,
    )
datasets = [*ruler_4k_datasets]

if "{thinking_prompt}" in os.environ.build()["SYSTEM_PROMPT_TEMPLATE"]:
    system_prompt = "thinking off"
else:
    system_prompt = ""

models[0]["system_prompt"] = system_prompt
