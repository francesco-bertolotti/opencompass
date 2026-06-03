# OpenCompass Compasses

A **compass** is a Python config file that tells OpenCompass which benchmarks to run and how to run them. This directory contains all compass files used by domyn-evals.

## How a compass is selected

In your main TOML config, set the `[opencompass] compass` field:

```toml
[opencompass]
compass = "main"   # resolves to main_think.py or main_nothink.py
```

The driver automatically appends `_think` or `_nothink` based on `[generation] reasoning`. You can also write the full name (e.g. `compass = "ruler_32k_think"`) to skip auto-detection — the driver will validate that the reasoning flag matches.

## Available compasses

| Compass (base name) | Contents |
|---|---|
| `main` | Core benchmarks: RULER-32k, MATH-500, HumanEval, LiveCodeBench, MGSM, MBPP |
| `full_{16k,32k}` | Full benchmark suite at 16k or 32k context |
| `mini_{16k}` | Smaller subset for quick runs |
| `ruler_{4k,8k,32k,64k,128k}` | RULER long-context benchmarks only |
| `mmlu` | MMLU only |
| `mmlu_pro` | MMLU-Pro only |
| `aime_pass48` | AIME 2024+2025 with pass@48 sampling |
| `domyn_large` | Large model benchmark set |
| `eval_cpt` | Continual pretraining evaluation |

Each base name has a `_think` and `_nothink` variant (except `eval_cpt`).

## Generation parameters

All sampling parameters are injected by the driver via environment variables — you do **not** edit them in the compass files:

| Env var | Source in main config |
|---|---|
| `TEMPERATURE` | `generation.think.temperature` / `generation.nothink.temperature` |
| `TOP_P` | `generation.think.top_p` / `generation.nothink.top_p` |
| `TOP_K` | `generation.think.top_k` |
| `MIN_P` | `generation.think.min_p` |
| `PRESENCE_PENALTY` | `generation.think.presence_penalty` |
| `REPETITION_PENALTY` | `generation.think.repetition_penalty` |
| `FREQUENCY_PENALTY` | `generation.think.frequency_penalty` |
| `MAX_TOKENS` | `generation.max_tokens` |
| `EXTRA_BODY` | `generation.chat_template_kwargs` (e.g. `enable_thinking`) |
| `SYSTEM_PROMPT_TEMPLATE` | `generation.think.system_prompt_template` |
| `SIZE_LIMIT` | `generation.max_samples` |
| `BATCH_SIZE` | `opencompass.batch_size` |
| `ENDPOINT` | `model.model_url` |
| `MODEL_PATH` | `model.model_path` |
| `DEBUG_INFERENCE` | `general.debug_inference` |

## How to create a custom compass

1. Create a new file, e.g. `my_eval_think.py` (always include `_think` or `_nothink` in the name).

2. Import the datasets you need using `read_base()`:
   ```python
   import os
   from mmengine.config import read_base

   with read_base():
       from opencompass.configs.datasets.math.math_500_gen import math_datasets
       from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets
       # or re-use an existing compass as a base:
       from .full_think_32k import models, infer, eval
   ```

3. Define the `datasets` list you want to run:
   ```python
   datasets = [
       *math_datasets,
       *gpqa_datasets,
   ]
   ```

4. Apply the size limit from env (required for `--max-samples` / `SIZE_LIMIT` to work):
   ```python
   size_limit = os.environ.build()["SIZE_LIMIT"]
   for dataset in datasets:
       dataset["reader_cfg"].setdefault("test_range", f"[slice(None,{size_limit},None)]")
   ```

5. Reference your new compass in the main config:
   ```toml
   [opencompass]
   compass = "my_eval"   # driver appends _think or _nothink automatically
   ```

If you need both thinking and non-thinking variants, create `my_eval_think.py` and `my_eval_nothink.py`. The non-thinking variant typically just imports from the thinking one and relies on the env vars for the difference (since generation params are all injected at runtime).
