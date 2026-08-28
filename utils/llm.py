from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .env import env_first, env_require


class LLM_Manager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        base = base_url or env_first("EVREN_BASE_URL", "EVREN_URL")
        if not base:
            raise RuntimeError("EVREN_BASE_URL veya EVREN_URL eksik. .env dosyasını kontrol et.")
        self.llm = ChatOpenAI(
            model=model_name or env_first("EVREN_LLM_MODEL", "LLM_NAME", default="llm-fast"),
            api_key=api_key or env_require("EVREN_API_KEY"),
            base_url=base,
            temperature=0.7,
        )

        self.history = []

        if system_prompt:
            self.history.append(SystemMessage(content=system_prompt))

    def run(self, query: str, *, raise_on_error: bool = False) -> str:
        self.history.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(self.history)
            self.history.append(AIMessage(content=response.content))
            return response.content
        except Exception as e:
            self.history.pop()
            if raise_on_error:
                raise
            return f"[LLM HATA] Model çağrısı başarısız oldu: {str(e)}"

    def clear_history(self):
        self.history = [msg for msg in self.history if isinstance(msg, SystemMessage)]
