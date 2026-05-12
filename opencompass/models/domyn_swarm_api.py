import opencompass
import diskcache
import traceback
import tenacity
import hashlib
import asyncio
import typing
import openai
import json
import os
from transformers import AutoTokenizer
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
        endpoint: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        extra_body: typing.Optional[typing.Dict[str, typing.Any]] = dict(),
        timeout: int = 1200,
        cache: str = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "opencompass_cache"
        ),
        max_tokens: int | None = None,
    ):
        super().__init__(path="", max_seq_len=max_tokens)
        self.system_prompt = system_prompt
        self.swarm_name = swarm_name
        self.temperature = temperature
        self.extra_body = extra_body
        self.max_tokens = max_tokens
        tokenizer_model = os.environ.get("TOKENIZER_MODEL")
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_model) if tokenizer_model else None
        )
        try:
            swarm = DomynLLMSwarm.from_state(self.swarm_name)
            self.endpoint = swarm.endpoint
            self.model = swarm.model
        except Exception:
            logger.error(
                f"Failed to initialize DomynSwarm with name {self.swarm_name}, trying to use provided endpoint {endpoint} and model {model}."
            )
            self.endpoint = endpoint
            self.model = model
            if not self.endpoint or not self.model:
                raise ValueError(
                    "Either swarm initialization must succeed or both endpoint and model must be provided."
                )
        if self.endpoint.endswith("v1/v1"):
            self.endpoint = self.endpoint.replace("/v1", "", 1)
        if not self.endpoint.endswith("/v1"):
            self.endpoint = self.endpoint.rstrip("/") + "/v1"

        self.cache = cache

        self.client = openai.AsyncOpenAI(
            base_url=f"{self.endpoint}",
            api_key="-",
            organization="-",
            project="-",
            timeout=timeout,
        )

    def generate(
        self,
        prompts: typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]],
        max_out_len: int = 512,
    ):
        return asyncio.run(self._generate(prompts, max_out_len))

    async def _generate(
        self,
        prompts: typing.List[typing.Union[opencompass.utils.prompt.PromptList, str]],
        max_out_len: int = 512,
    ) -> list[str]:

        @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=60, max=60),
            stop=tenacity.stop_after_attempt(180),
            retry=tenacity.retry_if_exception_type(
                (openai.APITimeoutError, openai.InternalServerError)
            ),
            reraise=True,
            before_sleep=lambda retry_state: print(
                f"Retrying due to timeout, attempt {retry_state.attempt_number}..."
            ),
        )
        async def complete(messages) -> list[str]:
            """Asynchronously complete the prompt using the OpenAI API."""
            try:
                with diskcache.Cache(self.cache) as cache:
                    if self.tokenizer is not None:
                        formatted = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        prompt_tokens = len(self.tokenizer.encode(formatted, add_special_tokens=False))
                    else:
                        prompt_tokens = (
                            sum(len(m.get("content", "")) for m in messages) // 4
                        )
                    available_tokens = max(
                        1, (self.max_seq_len or 32768) - prompt_tokens - 1
                    )
                    print(f"Prompt has approx {prompt_tokens} tokens.")
                    print(
                        f"Max seq len is {self.max_seq_len}, so available tokens for response is {available_tokens}."
                    )
                    request = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "extra_body": self.extra_body,
                        "max_tokens": min(
                            self.max_tokens or max_out_len, available_tokens
                        ),
                    }
                    print(
                        f"Request: {json.dumps(request['extra_body'], indent=2)}, max_tokens: {request['max_tokens']}"
                    )
                    key = hashlib.sha256(
                        json.dumps(request, sort_keys=True).encode()
                    ).hexdigest()

                    if key in cache:
                        response = cache[key]
                        # print("Cache hit")
                    else:
                        # print("Cache miss")
                        response = await self.client.chat.completions.create(**request)
                        try:
                            cache[key] = response
                        except OSError:
                            logger.warning(
                                "Failed to write to cache (disk full?), continuing without caching."
                            )
                print(f"Response: {response}")
                message = response.choices[0].message
                reasoning = getattr(message, "reasoning", None) or ""
                content = message.content or ""
                if reasoning:
                    return f"<think>{reasoning}</think>{content}"
                return content

            except openai.BadRequestError:
                traceback.print_exc()
                return ""

        return await asyncio.gather(
            *[complete(self.format(prompt)) for prompt in prompts]
        )

    def format(self, input: typing.Union[opencompass.utils.prompt.PromptList, str]):
        """Format the input into a message structure suitable for the API."""

        assert isinstance(input, typing.Union[opencompass.utils.prompt.PromptList, str])

        system_prompt = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

        if isinstance(input, str):
            messages = [*system_prompt, {"role": "user", "content": input}]
        else:
            messages = [*system_prompt]
            msg_buffer, last_role = [], None
            for item in input:
                item["role"] = "assistant" if item["role"] == "BOT" else "user"
                if item["role"] != last_role and last_role is not None:
                    messages.append(
                        {"content": "\n".join(msg_buffer), "role": last_role}
                    )
                    msg_buffer = []
                msg_buffer.append(item["prompt"])
                last_role = item["role"]
            messages.append({"content": "\n".join(msg_buffer), "role": last_role})

        return messages
