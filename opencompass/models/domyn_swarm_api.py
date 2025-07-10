import opencompass
import tenacity
import asyncio
import pathlib
import typing
import openai
import json
import os



class DomynSwarm(opencompass.models.base_api.BaseAPIModel):
    def __init__(
        self,
        state_path: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        extra_body: typing.Optional[typing.Dict[str, typing.Any]] = dict(),
        timeout: int = 1200,
    ):
        super().__init__(path="")
        self.system_prompt = system_prompt
        self.state_path = pathlib.Path(state_path)
        self.temperature = temperature
        self.extra_body = extra_body

        assert self.state_path.exists(), f"State file {self.state_path} does not exist. Please set SWARM_STATE environment variable to the path of the state file."

        self.swarm_state = json.load(self.state_path.open("r", encoding="utf-8"))
        self.endpoint = self.swarm_state["endpoint"]
        self.model = self.swarm_state["model"]
        self.client = openai.AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key="-",
            organization="-",
            project="-",
            timeout=timeout,
        )

    def generate(self, prompts : typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]], max_out_len: int = 512):
        return asyncio.run(self._generate(prompts, max_out_len))

    async def _generate(self, prompts : typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]], max_out_len: int = 512) -> list[str]:

        @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_exception_type(openai.APITimeoutError),
            reraise=True,
            before_sleep=lambda retry_state: print(f"Retrying due to timeout, attempt {retry_state.attempt_number}..."),
        )
        async def complete(messages) -> list[str]:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                extra_body=self.extra_body,
                max_tokens=max_out_len,
            )
            return resp.choices[0].message.content

        return await asyncio.gather(*[complete(self.format(prompt)) for prompt in prompts])

    def format(self, input:typing.Union[opencompass.utils.prompt.PromptList, str]):
        """ Format the input into a message structure suitable for the API. """

        assert isinstance(input, typing.Union[opencompass.utils.prompt.PromptList, str])

        system_prompt = [{"role" : "system", "content" : self.system_prompt}] if self.system_prompt else []

        if isinstance(input, str):
            messages = [*system_prompt, {"role": "user", "content": input}]
        else:
            messages = [*system_prompt]
            msg_buffer, last_role = [], None
            for item in input:
                item["role"] = "assistant" if item["role"] == "BOT" else "user"
                if item["role"] != last_role and last_role is not None:
                    messages.append({
                        "content": "\n".join(msg_buffer),
                        "role": last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item["prompt"])
                last_role = item["role"]
            messages.append({
                "content": "\n".join(msg_buffer),
                "role": last_role
            })

        return messages

