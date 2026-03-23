import os

from mmengine.config import read_base

with read_base():
    from .full_think_32k import models, infer, eval, ruler_32k_datasets  # noqa: F401

datasets = [*ruler_32k_datasets]

system_prompt = (
    os.environ.build()["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

models[0]["system_prompt"] = system_prompt
