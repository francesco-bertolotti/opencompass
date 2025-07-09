import opencompass
import pathlib
import typing
import pandas
import os

class Jsonl(opencompass.models.base_api.BaseAPIModel):
    def __init__(
        self,
        prompts_path: str,
        responses_path: str = None,
        system_prompt: str = None,
    ):
        super().__init__(path="")

        self.  prompts_path = pathlib.Path(os.path.expandvars(prompts_path))
        self.responses_path = pathlib.Path(os.path.expandvars(responses_path))
        self.logger = opencompass.utils.get_logger()

        self.system_prompt  = system_prompt
        print("responses_path", self.responses_path, bool(self.responses_path))

        self.  prompts_path.parent.mkdir(parents=True, exist_ok=True)
        self.responses_path.parent.mkdir(parents=True, exist_ok=True)

        self. eval_mode = responses_path and self.responses_path.exists()
        self.infer_mode = not self.eval_mode

        if responses_path and not self.responses_path.exists():
            self.logger.warning(f"Responses path {self.responses_path} provided but it does not exist.")

        if self.eval_mode:
            self.responses = iter(pandas.read_json(self.responses_path, lines=True, encoding='utf-8'))
            self.prompts   = iter(pandas.read_json(self.prompts_path  , lines=True, encoding='utf-8'))


    def generate(self, inputs : typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]], max_out_len: int = 512):
        """ Caches prompts and responses in JSONL format. Responses are empty strings if no responses_path is provided. """

        if self.infer_mode:
            prompts = pandas.DataFrame([self.format(input) for input in inputs])
            with open(self.prompts_path, 'a', encoding='utf-8') as f:
                prompts.to_json(f, orient="records", lines=True, force_ascii=False)
            return ["" for _ in inputs]

        if self.eval_mode:
            reponses = []
            for input in inputs:
                response   = next(self.responses)
                old_prompt = next(self.prompts)
                new_prompt = self.format(input)
                if str(new_prompt) != str(old_prompt): raise ValueError(f"Input prompt {input} does not match the cached prompt {old_prompt}. Please check the input format or the cached responses.")
                reponses.append(response["results"].split("</think>")[-1].strip())
            return reponses

    def format(self, input:typing.Union[opencompass.utils.prompt.PromptList, str]):
        """ Format the input into a message structure suitable for the API. """

        assert isinstance(input, typing.Union[opencompass.utils.prompt.PromptList, str])

        system_prompt = [{"role" : "system", "content" : self.system_prompt}] if self.system_prompt else []

        if isinstance(input, str):
            messages = [*system_prompt, {'role': 'user', 'content': input}]
        else:
            messages = [*system_prompt]
            msg_buffer, last_role = [], None
            for item in input:
                item['role'] = 'assistant' if item['role'] == 'BOT' else 'user'
                if item['role'] != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item['role']
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })

        return {"messages" : messages}
