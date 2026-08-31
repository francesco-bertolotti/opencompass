import os

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen import aime2024_datasets
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import (
        aime2025_datasets,
    )
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

    # from opencompass.configs.datasets.drop.drop_gen import drop_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.babilong.babilong_4k_gen import (
        babiLong_4k_datasets,
    )
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen import (
        LCB_datasets,
    )
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.mgsm.mgsm_gen import mgsm_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets

    # from opencompass.configs.datasets.summedits.summedits_gen import summedits_datasets
    from opencompass.configs.datasets.mmmlu_lite.mmmlu_lite_gen import (
        mmmlu_lite_datasets,
    )
    from opencompass.configs.datasets.ruler.ruler_32k_gen import (
        ruler_datasets as ruler_32k_datasets,
    )

    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups

# aime2025 disable llm as a judge
aime2025_datasets[0].eval_cfg.evaluator = dict(  # type: ignore
    type="opencompass.evaluator.MATHVerifyEvaluator"
)

datasets = [
    *aime2024_datasets,
    *aime2025_datasets,
    *mmlu_datasets,
    *gpqa_datasets,
    *gsm8k_datasets,
    # *drop_datasets,
    *humaneval_datasets,
    *babiLong_4k_datasets,
    *ifeval_datasets,
    *LCB_datasets,
    *math_datasets,
    *mbpp_datasets,
    *mgsm_datasets,
    *mmlu_pro_datasets,
    # *summedits_datasets,
    *mmmlu_lite_datasets,
    *ruler_32k_datasets,
]

size_limit = os.environ.build()["SIZE_LIMIT"]
for dataset in datasets:
    dataset["reader_cfg"].setdefault("test_range", f"[slice(None,{size_limit},None)]")

system_prompt = (
    os.environ.build()["SYSTEM_PROMPT_TEMPLATE"].replace("{instruction}", "").strip()
)

print("Extra body:", os.environ.build()["EXTRA_BODY"])
extra_body = (
    eval(
        os.environ.build()["EXTRA_BODY"]
        .replace("true", "True")
        .replace("false", "False")
    )
    if os.environ.build().get("EXTRA_BODY")
    else {}
)

models = [
    dict(
        type="opencompass.models.domyn_swarm_api.DomynSwarm",
        abbr=os.environ.build()["MODEL_ABBR"],
        batch_size=int(os.environ.build()["BATCH_SIZE"]),
        system_prompt=system_prompt,
        swarm_name=os.environ.build()["SWARM_NAME"],
        model=os.environ.build()["MODEL_PATH"],
        endpoint=os.environ.build()["ENDPOINT"],
        temperature=float(os.environ.build()["TEMPERATURE"]),
        extra_body={
            **dict(
                top_p=float(os.environ.build()["TOP_P"]),
                top_k=int(os.environ.build()["TOP_K"]),
                min_p=float(os.environ.build()["MIN_P"]),
                presence_penalty=float(os.environ.build()["PRESENCE_PENALTY"]),
                repetition_penalty=float(os.environ.build()["REPETITION_PENALTY"]),
                frequency_penalty=float(os.environ.build()["FREQUENCY_PENALTY"]),
            ),
            **extra_body,
        },
        max_tokens=int(os.environ.build()["MAX_TOKENS"]),
        # Job-private response cache. The DomynSwarm default is a single fixed
        # "$TMPDIR/opencompass_cache", and on Leonardo TMPDIR=/scratch_local is
        # node-local but world-writable and shared by every user on the node. Any
        # stale sqlite lock state left there — a -wal/-shm from someone else's
        # killed process, or a second run of ours on the same node — makes
        # diskcache.Cache() raise "sqlite3.OperationalError: locking protocol"
        # on EVERY request, and opencompass still writes a clean-looking summary
        # of dashes/zeros. That silently destroyed Ministral's rerun (all "-")
        # and nanbeige's duplicate. Scoping the path to the job id makes the
        # cache un-shareable, which is all we ever wanted from it.
        # Built with plain string concatenation on purpose: these compass
        # files are parsed by mmengine as LAZY configs, where `os` is a
        # LazyObject. os.environ.build() is fine (build() materialises it and
        # returns a real dict), but calling os.path.join() on a LazyObject
        # raises a bare RuntimeError at config-load time and the whole run
        # dies before it starts — while the driver still prints
        # "Task Complete".
        cache=os.environ.build().get("TMPDIR", "/tmp")
        + "/opencompass_cache_"
        + os.environ.build().get("SLURM_JOB_ID", "local"),
    )
]

infer = dict(
    partitioner=dict(
        type="opencompass.partitioners.NaivePartitioner",
    ),
    runner=dict(
        type="opencompass.runners.LocalRunner",
        task=dict(type="opencompass.tasks.openicl_infer.OpenICLInferTask"),
        max_num_workers=int(os.environ.build()["INFER_MAX_NUM_WORKERS"]),
    ),
)

eval = dict(
    partitioner=dict(
        type="opencompass.partitioners.NaivePartitioner",
    ),
    runner=dict(
        type="opencompass.runners.LocalRunner",
        task=dict(type="opencompass.tasks.openicl_eval.OpenICLEvalTask"),
        max_num_workers=int(os.environ.build()["EVAL_MAX_NUM_WORKERS"]),
    ),
)

summarizer = dict(
    dataset_abbrs=[
        "aime2024",
        "aime2025",
        "GPQA_diamond",
        "lukaemon_mmlu_college_biology",
        "lukaemon_mmlu_college_chemistry",
        "lukaemon_mmlu_college_computer_science",
        "lukaemon_mmlu_college_mathematics",
        "lukaemon_mmlu_college_physics",
        "lukaemon_mmlu_electrical_engineering",
        "lukaemon_mmlu_astronomy",
        "lukaemon_mmlu_anatomy",
        "lukaemon_mmlu_abstract_algebra",
        "lukaemon_mmlu_machine_learning",
        "lukaemon_mmlu_clinical_knowledge",
        "lukaemon_mmlu_global_facts",
        "lukaemon_mmlu_management",
        "lukaemon_mmlu_nutrition",
        "lukaemon_mmlu_marketing",
        "lukaemon_mmlu_professional_accounting",
        "lukaemon_mmlu_high_school_geography",
        "lukaemon_mmlu_international_law",
        "lukaemon_mmlu_moral_scenarios",
        "lukaemon_mmlu_computer_security",
        "lukaemon_mmlu_high_school_microeconomics",
        "lukaemon_mmlu_professional_law",
        "lukaemon_mmlu_medical_genetics",
        "lukaemon_mmlu_professional_psychology",
        "lukaemon_mmlu_jurisprudence",
        "lukaemon_mmlu_world_religions",
        "lukaemon_mmlu_philosophy",
        "lukaemon_mmlu_virology",
        "lukaemon_mmlu_high_school_chemistry",
        "lukaemon_mmlu_public_relations",
        "lukaemon_mmlu_high_school_macroeconomics",
        "lukaemon_mmlu_human_sexuality",
        "lukaemon_mmlu_elementary_mathematics",
        "lukaemon_mmlu_high_school_physics",
        "lukaemon_mmlu_high_school_computer_science",
        "lukaemon_mmlu_high_school_european_history",
        "lukaemon_mmlu_business_ethics",
        "lukaemon_mmlu_moral_disputes",
        "lukaemon_mmlu_high_school_statistics",
        "lukaemon_mmlu_miscellaneous",
        "lukaemon_mmlu_formal_logic",
        "lukaemon_mmlu_high_school_government_and_politics",
        "lukaemon_mmlu_prehistory",
        "lukaemon_mmlu_security_studies",
        "lukaemon_mmlu_high_school_biology",
        "lukaemon_mmlu_logical_fallacies",
        "lukaemon_mmlu_high_school_world_history",
        "lukaemon_mmlu_professional_medicine",
        "lukaemon_mmlu_high_school_mathematics",
        "lukaemon_mmlu_college_medicine",
        "lukaemon_mmlu_high_school_us_history",
        "lukaemon_mmlu_sociology",
        "lukaemon_mmlu_econometrics",
        "lukaemon_mmlu_high_school_psychology",
        "lukaemon_mmlu_human_aging",
        "lukaemon_mmlu_us_foreign_policy",
        "lukaemon_mmlu_conceptual_physics",
        # "drop",
        "gsm8k",
        "mmlu_pro_math",
        "mmlu_pro_physics",
        "mmlu_pro_chemistry",
        "mmlu_pro_law",
        "mmlu_pro_engineering",
        "mmlu_pro_other",
        "mmlu_pro_economics",
        "mmlu_pro_health",
        "mmlu_pro_psychology",
        "mmlu_pro_business",
        "mmlu_pro_biology",
        "mmlu_pro_philosophy",
        "mmlu_pro_computer_science",
        "mmlu_pro_history",
        "mgsm_bn",
        "mgsm_de",
        "mgsm_en",
        "mgsm_es",
        "mgsm_fr",
        "mgsm_ja",
        "mgsm_ru",
        "mgsm_sw",
        "mgsm_te",
        "mgsm_th",
        "mgsm_zh",
        ["mbpp", "score"],
        ["mbpp", "pass"],
        ["mbpp", "timeout"],
        ["mbpp", "failed"],
        ["mbpp", "wrong_answer"],
        "openai_humaneval",
        ["IFEval", "Prompt-level-strict-accuracy"],
        ["IFEval", "Inst-level-strict-accuracy"],
        ["IFEval", "Prompt-level-loose-accuracy"],
        ["IFEval", "Inst-level-loose-accuracy"],
        "math-500",
        "lcb_code_generation",
        "lcb_code_execution",
        "lcb_test_output",
        # "summedits",
        "mmlu-humanities",
        "mmlu-humanities",
        "mmlu-stem",
        "mmlu-stem",
        "mmlu-social-science",
        "mmlu-social-science",
        "mmlu-other",
        "mmlu-other",
        "mmlu",
        "mmlu",
        "mmlu-weighted",
        "mmlu-weighted",
        "mmlu_pro",
        "mmlu_pro",
        "mmlu_pro_weighted",
        "mmlu_pro_weighted",
        "babilong_qa1_4k",
        "babilong_qa2_4k",
        "babilong_qa3_4k",
        "babilong_qa4_4k",
        "babilong_qa5_4k",
        "babilong_qa6_4k",
        "babilong_qa7_4k",
        "babilong_qa8_4k",
        "babilong_qa9_4k",
        "openai_mmmlu_lite_AR-XY",
        "openai_mmmlu_lite_BN-BD",
        "openai_mmmlu_lite_DE-DE",
        "openai_mmmlu_lite_ES-LA",
        "openai_mmmlu_lite_FR-FR",
        "openai_mmmlu_lite_HI-IN",
        "openai_mmmlu_lite_ID-ID",
        "openai_mmmlu_lite_IT-IT",
        "openai_mmmlu_lite_JA-JP",
        "openai_mmmlu_lite_KO-KR",
        "openai_mmmlu_lite_PT-BR",
        "openai_mmmlu_lite_SW-KE",
        "openai_mmmlu_lite_YO-NG",
        "openai_mmmlu_lite_ZH-CN",
        "ruler_niah_single_1_32k",
        "ruler_niah_single_2_32k",
        "ruler_niah_single_3_32k",
        "ruler_niah_multikey_1_32k",
        "ruler_niah_multikey_2_32k",
        "ruler_niah_multikey_3_32k",
        "ruler_niah_multivalue_32k",
        "ruler_niah_multiquery_32k",
        "ruler_vt_32k",
        "ruler_fwe_32k",
        "ruler_cwe_32k",
        "ruler_qa_squad_32k",
        "ruler_qa_hotpotqa_32k",
        "ruler_32k",
    ],
    summary_groups=[
        *mmlu_summary_groups,
        *mmlu_pro_summary_groups,
        {
            "name": "mmlu_pro_weighted",
            "subsets": mmlu_pro_summary_groups[0]["subsets"],
            "weights": {
                "mmlu_pro_math": 11.2,
                "mmlu_pro_physics": 10.8,
                "mmlu_pro_chemistry": 9.4,
                "mmlu_pro_law": 9.2,
                "mmlu_pro_engineering": 8.1,
                "mmlu_pro_other": 7.7,
                "mmlu_pro_economics": 7,
                "mmlu_pro_health": 6.8,
                "mmlu_pro_psychology": 6.6,
                "mmlu_pro_business": 6.6,
                "mmlu_pro_biology": 6,
                "mmlu_pro_philosophy": 4.1,
                "mmlu_pro_computer_science": 3.4,
                "mmlu_pro_history": 3.2,
            },
        },
        ruler_summary_groups[3],  # 32k
    ],
)
