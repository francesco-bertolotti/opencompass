from opencompass.datasets import (
    EUMathOlympDeu,
    EUMathOlympEsp,
    EUMathOlympFra,
    EUMathOlympIta,
)
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

eu_math_olymp_reader_cfg = dict(input_columns=["question"], output_column="answer")

eu_math_olymp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2024_eval_cfg = dict(evaluator=dict(type=MATHVerifyEvaluator))
eu_math_olymp_datasets = [
    dict(
        abbr="eu_math_olymp_ita",
        type=EUMathOlympIta,
        path="opencompass/eu_math_olymp.parquet",
        reader_cfg=eu_math_olymp_reader_cfg,
        infer_cfg=eu_math_olymp_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    ),
    dict(
        abbr="eu_math_olymp_esp",
        type=EUMathOlympEsp,
        path="opencompass/eu_math_olymp.parquet",
        reader_cfg=eu_math_olymp_reader_cfg,
        infer_cfg=eu_math_olymp_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    ),
    dict(
        abbr="eu_math_olymp_fra",
        type=EUMathOlympFra,
        path="opencompass/eu_math_olymp.parquet",
        reader_cfg=eu_math_olymp_reader_cfg,
        infer_cfg=eu_math_olymp_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    ),
    dict(
        abbr="eu_math_olymp_deu",
        type=EUMathOlympDeu,
        path="opencompass/eu_math_olymp.parquet",
        reader_cfg=eu_math_olymp_reader_cfg,
        infer_cfg=eu_math_olymp_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    ),
]
