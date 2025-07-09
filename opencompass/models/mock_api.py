import opencompass
import typing

class Mock(opencompass.models.base_api.BaseAPIModel):
    def __init__(
        self,
    ):
        super().__init__(path="")

    def generate(self, prompts : typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]], max_out_len: int = 512):
        return ["\\boxed{3}" for _ in prompts]

