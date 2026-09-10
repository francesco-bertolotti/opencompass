import os

from mmengine.config import read_base  # type: ignore

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen import (
        LCB_datasets,
    )
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

datasets = [
    *humaneval_datasets,
    *LCB_datasets,
    *mbpp_datasets,
    *math_datasets,
]

_env = os.environ.build() # type: ignore

size_limit = _env["SIZE_LIMIT"]
for dataset in datasets:
    dataset["reader_cfg"].setdefault("test_range", f"[slice(None,{size_limit},None)]")

system_prompt = (
    _env["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

extra_body = (
    eval(
        _env["EXTRA_BODY"]
        .replace("true", "True")
        .replace("false", "False")
    )
    if _env.get("EXTRA_BODY")
    else {}
)

_sampling_raw = {
    "top_p": _env["TOP_P"],
    "top_k": _env["TOP_K"],
    "min_p": _env["MIN_P"],
    "presence_penalty": _env["PRESENCE_PENALTY"],
    "repetition_penalty": _env["REPETITION_PENALTY"],
    "frequency_penalty": _env["FREQUENCY_PENALTY"],
}
sampling_params = {
    key: (int(value) if key == "top_k" else float(value))
    for key, value in _sampling_raw.items()
    if value not in ("", "None")
}
temperature = (
    None if _env["TEMPERATURE"] in ("", "None") else float(_env["TEMPERATURE"])
)

models = [
    {
        "type": "opencompass.models.domyn_swarm_api.DomynSwarm",
        "abbr": _env["MODEL_ABBR"],
        "batch_size": int(_env["BATCH_SIZE"]),
        "system_prompt": system_prompt,
        "swarm_name": _env["SWARM_NAME"],
        "model": _env["MODEL_PATH"],
        "endpoint": _env["ENDPOINT"],
        "temperature": temperature,
        "extra_body": {**sampling_params, **extra_body},
        "max_tokens": int(_env["MAX_TOKENS"]),
        "cache": _env.get("TMPDIR", "/tmp")
        + "/opencompass_cache_"
        + _env.get("SLURM_JOB_ID", "local"),
    }
]

infer = {
    "partitioner": {
        "type": "opencompass.partitioners.NaivePartitioner",
    },
    "runner": {
        "type": "opencompass.runners.LocalRunner",
        "task": {"type": "opencompass.tasks.openicl_infer.OpenICLInferTask"},
        "max_num_workers": int(_env["INFER_MAX_NUM_WORKERS"]),
    },
}

eval = {
    "partitioner": {
        "type": "opencompass.partitioners.NaivePartitioner",
    },
    "runner": {
        "type": "opencompass.runners.LocalRunner",
        "task": {"type": "opencompass.tasks.openicl_eval.OpenICLEvalTask"},
        "max_num_workers": int(_env["EVAL_MAX_NUM_WORKERS"]),
    },
}

summarizer = {
    "dataset_abbrs": [
        ["mbpp", "score"],
        ["mbpp", "pass"],
        ["mbpp", "timeout"],
        ["mbpp", "failed"],
        ["mbpp", "wrong_answer"],
        "openai_humaneval",
        "math-500",
        "lcb_code_generation",
        "lcb_code_execution",
        "lcb_test_output",
    ],
}
