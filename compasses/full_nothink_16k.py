import os

from mmengine.config import read_base

with read_base():
    from .full_think_16k import datasets, models, infer, eval, summarizer  # noqa: F401

system_prompt = (
    os.environ.build()["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

models[0]["system_prompt"] = system_prompt
# models[0]["extra_body"]["max_tokens"] = 16384
