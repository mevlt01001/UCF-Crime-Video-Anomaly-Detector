from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .env import env_get, env_require


class LLM_Manager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = ChatOpenAI(
            model=model_name or env_get("EVREN_LLM_MODEL", "llm-fast"),
            api_key=api_key or env_require("EVREN_API_KEY"),
            base_url=base_url or env_require("EVREN_BASE_URL"),
            temperature=0.7,
        )

        self.history = []

        if system_prompt:
            self.history.append(SystemMessage(content=system_prompt))

    def run(self, query: str) -> str:
        self.history.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(self.history)
            self.history.append(AIMessage(content=response.content))
            return response.content
        except Exception as e:
            self.history.pop()
            return f"[LLM HATA] Model çağrısı başarısız oldu: {str(e)}"

    def clear_history(self):
        self.history = [msg for msg in self.history if isinstance(msg, SystemMessage)]
