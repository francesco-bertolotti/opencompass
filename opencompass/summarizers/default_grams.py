# flake8: noqa
# yapf: disable
import getpass
import os.path as osp
from datetime import datetime

from .default import DefaultSummarizer

import tabulate


class DefaultSummarizerGrams(DefaultSummarizer):
    """Default summarizer in OpenCompass.

    Args:
        config (ConfigDict): The configuration object of the evaluation task. It's expected to be filled out at runtime.
        dataset_abbrs (list[str], optional): Dataset abbreviations to be listed in the summary.
        summary_groups (list): The dataset groups whose results need to be averaged out. For example, mmlu. Each item it a dict with
            'name' (str) and 'subsets' (list of dataset abbrs), and optionally
            'weights' if weighted average is needed.
        prompt_db: A deprecated field.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):  # noqa

        # pick up results
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = self._pick_up_results()

        # calculate group metrics
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            self._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # format table
        table = self._format_table(parsed_results, dataset_metrics, dataset_eval_mode)

        # format raw txt
        raw_txts = self._format_raw_txt(raw_results)

        # output to screen
        print(tabulate.tabulate(table, headers='firstrow', floatfmt='.2f'))

        # output to .text / .csv files
        self._output_to_file(output_path, time_str, table, raw_txts)

        if self.lark_reporter:
            content = f'{getpass.getuser()} 的'
            content += f'详细评测汇总已输出至 {osp.abspath(output_path)}'
            self.lark_reporter.post(content)