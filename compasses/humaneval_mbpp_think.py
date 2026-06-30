from mmengine.config import read_base

with read_base():
    # Reuse the proven model / inference / eval runner config (think, 32k).
    # `size_limit` is computed there from $SIZE_LIMIT (--max-samples).
    from .full_think_32k import models, infer, eval, size_limit  # noqa: F401
    # pass@1 dataset variants (local code execution — no Docker, no remote API).
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import (
        humaneval_datasets,
    )
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_830460 import (
        sanitized_mbpp_datasets,
    )

# NOTE: pass@1 only. Real pass@k>1 is NOT possible with the DomynSwarm API model
# as-is: setting dataset["n"]=K is read only by the evaluator, but the model
# returns a single completion per prompt (it sends no `n` to the endpoint and
# uses choices[0]), and its response cache would make repeated-dataset sampling
# return identical completions. Generating K samples requires changes to
# opencompass/models/domyn_swarm_api.py (send n=K, return all choices, include
# n in the cache key). Until then, keep k=[1] / no `n`.

humaneval = humaneval_datasets[0]
# HumanEvalEvaluator: with one sample per problem, only pass@1 is well-defined.
humaneval["eval_cfg"]["k"] = [1]

# Sanitized MBPP — plain MBPPEvaluator emits a single pass score (no k needed).
mbpp = sanitized_mbpp_datasets[0]

datasets = [humaneval, mbpp]

# Honor SIZE_LIMIT (--max-samples) like the other compasses.
# WARNING: HumanEval's evaluator (evaluate_functional_correctness) asserts that
# ALL canonical problems are attempted, so --max-samples will make the HumanEval
# task fail with "Some problems are not attempted". Run the full set for HumanEval;
# only use --max-samples for MBPP-only smoke tests.
for dataset in datasets:
    dataset["reader_cfg"].setdefault("test_range", f"[slice(None,{size_limit},None)]")
