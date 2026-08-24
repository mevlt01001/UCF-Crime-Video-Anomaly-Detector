import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class LLM_Manager:
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model_name: str = "llm-fast", 
                 system_prompt: str = None):

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7
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

