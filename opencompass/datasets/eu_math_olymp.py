import pandas as pd
from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


def read_dataset(path: str, lang: str) -> list[dict[str, str]]:
    df = pd.read_parquet(path)
    df = df[df["language"] == lang]
    df = df[["question", "answer"]]
    return df.to_dict(orient="records")


@LOAD_DATASET.register_module()
class EUMathOlympIta(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = read_dataset(path, lang="ita")
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class EUMathOlympFra(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = read_dataset(path, lang="fra")
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class EUMathOlympDeu(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = read_dataset(path, lang="deu")
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class EUMathOlympEsp(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = read_dataset(path, lang="esp")
        return Dataset.from_list(dataset)
