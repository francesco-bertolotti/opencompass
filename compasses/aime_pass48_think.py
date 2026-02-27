from mmengine.config import read_base

with read_base():
    from .full_think_32k import models, infer, eval  # noqa: F401
    from .full_think_32k import aime2024_datasets, aime2025_datasets


aime24 = aime2024_datasets[0]
aime24["n"] = 48

aime25 = aime2025_datasets[0]
aime25["n"] = 48

datasets = [aime24, aime25]
