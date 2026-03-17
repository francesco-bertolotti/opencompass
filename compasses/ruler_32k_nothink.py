import os

from mmengine.config import read_base

with read_base():
    from .full_think_32k import models, infer, eval, ruler_32k_datasets  # noqa: F401

datasets = [*ruler_32k_datasets]

if "{thinking_prompt}" in os.environ.build()["SYSTEM_PROMPT_TEMPLATE"]:
    system_prompt = "thinking off"
else:
    system_prompt = ""

models[0]["system_prompt"] = system_prompt
