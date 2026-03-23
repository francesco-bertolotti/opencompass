import os

from mmengine.config import read_base

with read_base():
    from .full_think_16k import (
        models,
        infer,  # noqa: F401
        eval,  # noqa: F401
        aime2024_datasets,
        aime2025_datasets,
        gpqa_datasets,
    )


aime24 = aime2024_datasets[0]
aime24["n"] = 1

aime25 = aime2025_datasets[0]
aime25["n"] = 1

gpqa = gpqa_datasets[0]
gpqa["n"] = 1

datasets = [aime24, aime25, gpqa]

system_prompt = (
    os.environ.build()["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

models[0]["system_prompt"] = system_prompt
# models[0]["extra_body"]["max_tokens"] = 16384
