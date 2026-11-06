import hashlib
import os
import threading
import time
import warnings
from typing import Protocol


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIChatClient:
    def __init__(self, llm_cfg):
        import openai
        key = os.environ.get(llm_cfg.api_key_env)
        if not key:
            raise ValueError(f'Missing API key in environment variable {llm_cfg.api_key_env}')
        self.cfg = llm_cfg
        self.client = openai.OpenAI(api_key=key, base_url=llm_cfg.base_url or None, max_retries=0)

    def complete(self, system: str, user: str) -> str:
        import openai
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.name, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p, max_tokens=self.cfg.max_tokens)
                choice = response.choices[0]
                content = choice.message.content
                if content and content.strip():
                    if choice.finish_reason == 'length':
                        warnings.warn('LLM output truncated by llm.max_tokens', stacklevel=2)
                    return content
                if attempt == self.cfg.max_retries:
                    raise RuntimeError('Empty LLM response after retries')
                time.sleep(min(2 ** attempt, 60))
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError) as exc:
                retryable = not isinstance(exc, openai.APIStatusError) or exc.status_code in (408, 409, 429) or exc.status_code >= 500
                if not retryable or attempt == self.cfg.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 60))


class FakeChatClient:
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        with self._lock:
            self.calls += 1
        digest = hashlib.sha256(user.encode('utf-8')).hexdigest()[:6]
        return f'Fake description {digest}: {user[:60]}'


def make_chat_client(llm_cfg):
    if llm_cfg.provider == 'openai':
        return OpenAIChatClient(llm_cfg)
    if llm_cfg.provider == 'fake':
        return FakeChatClient()
    raise ValueError(f'Unknown LLM provider: {llm_cfg.provider}')
