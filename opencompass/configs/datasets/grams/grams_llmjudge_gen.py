from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GramsDatasetIt, GramsDatasetEs, GramsDatasetFr, GramsDatasetPt, GramsDatasetDe
from opencompass.evaluator import DomynSwarmLLMEvaluator
from opencompass.datasets import grams_llmjudge_postprocess

grams_reader_cfg = dict(input_columns=['problem'], output_column='language')

grams_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


GRADER_TEMPLATE = """
### Task:

{problem}

### Answer: 

{prediction}

Please provide the scores following this format:

{
  "Grammar": <score>,
  "Creativity": <score>,
  "Coherence": <score>,
  "Meaningfulness": <score>,
  "Lexical Richness": <score>
}

Respond only with the JSON object, without any additional text or explanation.
""".strip()


SYSTEM_PROMPT = """You are a strict and consistent evaluator.

You will be given a task (a question, instruction, or prompt) and an answer written in one of the following languages: Italian, French, Portuguese, Spanish, or German.
Your role is to evaluate the answer only, based on the quality dimensions below. The evaluation must be language-agnostic: apply the same criteria regardless of which of the five languages is used.

Input Format

The input will always be structured as follows:

### Task:

<text in Italian/French/Portuguese/Spanish/German> 

### Answer:

<text in the same language as the task>

You must ignore the task text when scoring; it is only provided so you understand what the answer responds to.
All scoring must apply exclusively to the answer.

Evaluation Dimensions and Rubric

Grammar (1–5)

5 = Perfect, native-like, no errors, no anglicisms.

4 = Few slips, errors are irrelevant and do not hinder understanding.

3 = Noticeable, some errors or awkward anglicisms, but understandable.

2 = Frequent mistakes, visible English influence; readability suffers.

1 = Severe, major grammatical problems; difficult to understand.

Creativity (1–5)

5 = Highly creative, original, engaging, strong stylistic flair.

4 = Somewhat creative, with interesting ideas or phrasing.

3 = Average, somewhat generic, limited originality.

2 = Mostly repetitive or cliché, lacking originality.

1 = Very banal or formulaic, no sign of creativity.

Coherence (1–5)

5 = Perfectly coherent; ideas well connected, no contradictions.

4 = Mostly coherent; minor lapses but overall clear.

3 = Some issues with flow or consistency, still understandable.

2 = Weak coherence; poorly connected or contradictory.

1 = Incoherent or confusing; no logical flow.

Meaningfulness / Appropriateness (1–5)

5 = Fully meaningful, relevant, directly addresses the task.

4 = Mostly meaningful and relevant, slight digressions.

3 = Somewhat meaningful; partially relevant, may miss key aspects.

2 = Largely irrelevant or nonsensical in parts.

1 = Completely irrelevant, nonsensical, or off-topic.

Lexical Richness (1–5)

5 = Rich, varied, precise vocabulary; highly appropriate word choice.

4 = Good vocabulary use, some variety, not very sophisticated.

3 = Adequate vocabulary, somewhat repetitive or basic.

2 = Limited vocabulary, simplistic or occasionally inappropriate.

1 = Very poor lexical range; repetitive, inappropriate, or incorrect usage.

Output Format

Return the evaluation strictly in JSON format:

{
  "Grammar": <score>,
  "Creativity": <score>,
  "Coherence": <score>,
  "Meaningfulness": <score>,
  "Lexical Richness": <score>
}

Do not provide explanations or additional commentary unless explicitly requested.
""".strip()

grams_eval_cfg_it = dict(
    evaluator=dict(
        type=DomynSwarmLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=SYSTEM_PROMPT,
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=GramsDatasetIt,
            path='opencompass/grams_it',
            reader_cfg=grams_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=grams_llmjudge_postprocess),
    )
)

grams_eval_cfg_es = dict(
    evaluator=dict(
        type=DomynSwarmLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=SYSTEM_PROMPT,
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=GramsDatasetEs,
            path='opencompass/grams_es',
            reader_cfg=grams_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=grams_llmjudge_postprocess),
    )
)

grams_eval_cfg_fr = dict(
    evaluator=dict(
        type=DomynSwarmLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=SYSTEM_PROMPT,
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=GramsDatasetFr,
            path='opencompass/grams_fr',
            reader_cfg=grams_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=grams_llmjudge_postprocess),
    )
)

grams_eval_cfg_de = dict(
    evaluator=dict(
        type=DomynSwarmLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=SYSTEM_PROMPT,
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=GramsDatasetDe,
            path='opencompass/grams_de',
            reader_cfg=grams_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=grams_llmjudge_postprocess),
    )
)

grams_eval_cfg_pt = dict(
    evaluator=dict(
        type=DomynSwarmLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt=SYSTEM_PROMPT,
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=GramsDatasetPt,
            path='opencompass/grams_pt',
            reader_cfg=grams_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=grams_llmjudge_postprocess),
    )
)

grams_datasets = [
    dict(
        abbr='grams_it',
        type=GramsDatasetIt,
        path='opencompass/grams_it',
        reader_cfg=grams_reader_cfg,
        infer_cfg=grams_infer_cfg,
        eval_cfg=grams_eval_cfg_it,
    ),
    dict(
        abbr='grams_es',
        type=GramsDatasetEs,
        path='opencompass/grams_es',
        reader_cfg=grams_reader_cfg,
        infer_cfg=grams_infer_cfg,
        eval_cfg=grams_eval_cfg_es,
    ),
    dict(
        abbr='grams_fr',
        type=GramsDatasetFr,
        path='opencompass/grams_fr',
        reader_cfg=grams_reader_cfg,
        infer_cfg=grams_infer_cfg,
        eval_cfg=grams_eval_cfg_fr,
    ),
    dict(
        abbr='grams_de',
        type=GramsDatasetDe,
        path='opencompass/grams_de',
        reader_cfg=grams_reader_cfg,
        infer_cfg=grams_infer_cfg,
        eval_cfg=grams_eval_cfg_de,
    ),
    dict(
        abbr='grams_pt',
        type=GramsDatasetPt,
        path='opencompass/grams_pt',
        reader_cfg=grams_reader_cfg,
        infer_cfg=grams_infer_cfg,
        eval_cfg=grams_eval_cfg_pt,
    )
]
