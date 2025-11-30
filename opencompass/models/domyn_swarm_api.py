import opencompass
import diskcache
import traceback
import tenacity
import hashlib
import asyncio
import typing
import openai
import json
from domyn_swarm import DomynLLMSwarm

from opencompass.registry import MODELS
from .base_api import BaseAPIModel

from ..utils.logging import get_logger
logger = get_logger(__name__)


@MODELS.register_module()
class DomynSwarm(BaseAPIModel):
    def __init__(
        self,
        swarm_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        extra_body: typing.Optional[typing.Dict[str, typing.Any]] = dict(),
        timeout: int = 1200,
        cache: str = "/tmp/opencompass_cache",
    ):
        super().__init__(path="")
        self.system_prompt = system_prompt
        self.swarm_name = swarm_name
        self.temperature = temperature
        self.extra_body = extra_body

        swarm = DomynLLMSwarm.from_state(self.swarm_name)
        self.endpoint = swarm.endpoint
        self.model = swarm.model
        self.cache = cache

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
            wait=tenacity.wait_exponential(multiplier=1, min=60, max=60),
            stop=tenacity.stop_after_attempt(180),
            retry=tenacity.retry_if_exception_type((openai.APITimeoutError, openai.InternalServerError)),
            reraise=True,
            before_sleep=lambda retry_state: print(f"Retrying due to timeout, attempt {retry_state.attempt_number}..."),
        )
        async def complete(messages) -> list[str]:
            """ Asynchronously complete the prompt using the OpenAI API. """
            try:
                with diskcache.Cache(self.cache) as cache:
                    request = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "extra_body": self.extra_body,
                    }
                    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

                    if key in cache:
                        response = cache[key]
                        print("Cache hit")
                    else:
                        print("Cache miss")
                        response = await self.client.chat.completions.create(**request)
                        cache[key] = response

                    
                return response.choices[0].message.content
            except openai.BadRequestError:
                traceback.print_exc()
                return ""

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

