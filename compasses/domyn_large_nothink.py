import os
from mmengine.config import read_base

with read_base():
    from .domyn_large_think import models, datasets, infer, eval, summarizer  # noqa: F401

if "{thinking_prompt}" in os.environ.build()["SYSTEM_PROMPT_TEMPLATE"]:
    system_prompt = "thinking off"
else:
    system_prompt = ""

models[0]["system_prompt"] = system_prompt
