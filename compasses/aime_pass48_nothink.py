import os

from mmengine.config import read_base

with read_base():
    from .full_think_32k import models, infer, eval  # noqa: F401
    from .full_think_32k import aime2024_datasets, aime2025_datasets


aime24 = aime2024_datasets[0]
aime24["n"] = 48

aime25 = aime2025_datasets[0]
aime25["n"] = 48

datasets = [aime24, aime25]

if "{thinking_prompt}" in os.environ.build()["SYSTEM_PROMPT_TEMPLATE"]:
    system_prompt = "thinking off"
else:
    system_prompt = ""

models[0]["system_prompt"] = system_prompt
# models[0]["extra_body"]["max_tokens"] = 32768
