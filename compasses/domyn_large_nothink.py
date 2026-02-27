from mmengine.config import read_base

with read_base():
    from .domyn_large_think import models, datasets, infer, eval, summarizer  # noqa: F401

models[0]["system_prompt"] = "thinking off"
