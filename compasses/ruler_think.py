from mmengine.config import read_base # type: ignore

with read_base():
    from .opencompass_think import models, infer, eval
    from opencompass.configs.datasets.ruler.ruler_512k_gen import (
        ruler_datasets as ruler_512k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_256k_gen import (
        ruler_datasets as ruler_256k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_128k_gen import (
        ruler_datasets as ruler_128k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_64k_gen import (
        ruler_datasets as ruler_64k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_32k_gen import (
        ruler_datasets as ruler_32k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_16k_gen import (
        ruler_datasets as ruler_16k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_8k_gen import (
        ruler_datasets as ruler_8k_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_4k_gen import (
        ruler_datasets as ruler_4k_datasets,
    )
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups


datasets = [
    *ruler_512k_datasets, 
    *ruler_256k_datasets, 
    *ruler_128k_datasets, 
    *ruler_64k_datasets, 
    *ruler_32k_datasets, 
    *ruler_16k_datasets, 
    *ruler_8k_datasets, 
    *ruler_4k_datasets
    ]

summarizer = {
    "dataset_abbrs": [
        "ruler_512k",
        "ruler_256k",
        "ruler_128k",
        "ruler_64k",
        "ruler_32k",
        "ruler_16k",
        "ruler_8k",
        "ruler_4k",
    ],
    "summary_groups": ruler_summary_groups,
}
