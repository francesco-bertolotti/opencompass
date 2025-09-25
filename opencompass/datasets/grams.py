import json

import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path, get_logger

from .base import BaseDataset

from typing import Dict, Optional

EXPECTED_KEYS = {
    "Grammar", "Creativity", "Coherence", "Meaningfulness", "Lexical Richness"
}

SAMPLE_LIMITS = 500

TEMPLATE_IT = """
Rispondi in italiano alla richiesta usando il testo fornito.

Testo: {context}
Richiesta: {question}
"""

TEMPLATE_ES = """
Responda en español a la solicitud utilizando el texto proporcionado.

Texto: {context}
Solicitud: {question}
"""

TEMPLATE_PT = """
Responda em português à solicitação usando o texto fornecido.

Texto: {context}
Solicitação: {question}
"""

TEMPLATE_FR = """
Répondez en français à la demande en utilisant le texte fourni.

Texte: {context}
Demande: {question}
"""

TEMPLATE_DE = """
Beantworte die Anfrage auf Deutsch unter Verwendung des bereitgestellten Textes.

Text: {context}
Anfrage: {question}
"""

@LOAD_DATASET.register_module()
class GramsDatasetIt(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['problem'] = TEMPLATE_IT.format(context=line['context'], question=line['question'])
                dataset.append(line)
        dataset = dataset[:SAMPLE_LIMITS]
        return Dataset.from_list(dataset)

@LOAD_DATASET.register_module()
class GramsDatasetFr(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['problem'] = TEMPLATE_FR.format(context=line['context'], question=line['question'])
                dataset.append(line)
        dataset = dataset[:SAMPLE_LIMITS]
        return Dataset.from_list(dataset)

@LOAD_DATASET.register_module()
class GramsDatasetDe(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['problem'] = TEMPLATE_DE.format(context=line['context'], question=line['question'])
                dataset.append(line)
        dataset = dataset[:SAMPLE_LIMITS]
        return Dataset.from_list(dataset)
    
@LOAD_DATASET.register_module()
class GramsDatasetEs(BaseDataset):
    
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['problem'] = TEMPLATE_ES.format(context=line['context'], question=line['question'])
                dataset.append(line)
        dataset = dataset[:SAMPLE_LIMITS]
        return Dataset.from_list(dataset)
    
@LOAD_DATASET.register_module()
class GramsDatasetPt(BaseDataset):
    
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['problem'] = TEMPLATE_PT.format(context=line['context'], question=line['question'])
                dataset.append(line)
        dataset = dataset[:SAMPLE_LIMITS]
        return Dataset.from_list(dataset)


def _get_grams_final_results(judged_answers, metric_name='score'):
    """
    Goal of this function is to provide the average score for each of the five dimensions
    evaluated by the LLM judge, along with some additional statistics.
    For each dimension, we compute:
    - Mean Score: The average score across all judged answers.
    - Count of Valid Scores: The number of answers that received a valid score (not None).
    - Count of Missing Scores: The number of answers that did not receive a valid score (None).

    If an answer has None for all dimensions, it is considered as not attempted.
    So we also compute:
    - Not Attempted Count: The number of answers that did not receive any valid scores.
    - Attempted Count: The number of answers that received at least one valid score.
    """

    is_not_attempted_count = 0
    attempted_judge_count = 0

    scores = {key: {'total': 0, 'valid_count': 0, 'missing': 0} for key in EXPECTED_KEYS}

    for judged_answer in judged_answers:
        if judged_answer is None:
            is_not_attempted_count += 1
        else:
            attempted_judge_count += 1
            for k, v in judged_answer.items():
                if v is None:
                    scores[k]['missing'] += 1
                else:
                    scores[k]['total'] += v
                    scores[k]['valid_count'] += 1

    means = {
        f"{metric_name}_{k}": (v['total'] / v['valid_count'] if v['valid_count'] > 0 else 0)
        for k, v in scores.items()
    }

    valid_counts = {
        f"valid_count_ratio_{k}": v['valid_count'] / (v['valid_count'] + v['missing'] ) if (v['valid_count'] + v['missing']) > 0 else 0
        for k, v in scores.items()
    }

    attempted_judge_ratio = attempted_judge_count / (attempted_judge_count + is_not_attempted_count)

    result = {
        **means,
        **valid_counts,
        'attempted_ratio': attempted_judge_ratio,
    }
    return result


def _grams_llmjudge_postprocess(judgement: str) -> Optional[Dict[str, float]]:
    """
    LLM judge output format example should be a json object like this:
    {
        "Grammar": <score>,
        "Creativity": <score>,
        "Coherence": <score>,
        "Meaningfulness": <score>,
        "Lexical Richness": <score>
    }
    """
    try:
        # Extract JSON substring if extra text is included
        match = re.search(r"\{.*\}", judgement, re.DOTALL)
        if not match:
            get_logger().warning("No JSON object found in judgement.")
            return None
        
        judgement_json = json.loads(match.group(0))

        if not isinstance(judgement_json, dict):
            get_logger().warning("Judgement is not a valid JSON object.")
            return None
        
        # Normalize keys
        normalized = {k.strip(): v for k, v in judgement_json.items()}
        
        # Ensure all expected keys exist
        for key in EXPECTED_KEYS:
            if key not in normalized:
                get_logger().warning(f"Missing key: {key}")
                normalized[key] = None
        
        # Coerce values to floats
        for key in EXPECTED_KEYS:
            try:
                normalized[key] = float(normalized[key])
            except (ValueError, TypeError):
                get_logger().warning(f"Invalid score for {key}, setting to None.")
                normalized[key] = None

        return normalized
    except Exception as e:
        get_logger().warning(f"Postprocess failed: {e}")
        return None


def grams_llmjudge_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    for k, v in output.items():
        processed_judge = _grams_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
    results = _get_grams_final_results(judged_answers)
    results['details'] = output
    return results
