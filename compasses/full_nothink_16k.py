from mmengine.config import read_base

with read_base():
    from .full_think_16k import datasets, models, infer, eval, summarizer  # noqa: F401

models[0]["system_prompt"] = "thinking off"
# models[0]["extra_body"]["max_tokens"] = 16384
