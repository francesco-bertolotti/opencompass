from mmengine.config import read_base

with read_base():
    from .full_think_16k import (
        models,  # noqa: F401
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
