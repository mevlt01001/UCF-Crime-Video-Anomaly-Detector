import os

from typing import TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from utils import tools, LLM_Manager
from utils import (
    build_planner_system_prompt,
    build_executor_system_prompt,
    build_reviewer_system_prompt
)

from dotenv import load_dotenv
load_dotenv()

EVREN_API_KEY = os.environ.get("EVREN_API_KEY")
EVREN_URL = os.environ.get("EVREN_URL")
LLM_SYSTEM_PROMT = os.environ.get("LLM_SYSTEM_PROMT")

llm = ChatOpenAI(
    model="llm-fast",
    api_key=EVREN_API_KEY, 
    base_url=EVREN_URL, 
    temperature=0.0 
)

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    user_query: str
    video_paths: list[str]
    image_paths: list[str]
    plan: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    feedback: str
    final_answer: str

class ReviewResult(BaseModel):
    is_complete: bool = Field(description="İşlemcinin bulguları kullanıcının isteğini tam olarak karşılıyor mu?")
    feedback_or_answer: str = Field(description="Eğer eksik varsa İşlemciye verilecek geri bildirim. Tamamsa kullanıcıya sunulacak nihai, derlenmiş cevap.")

def planner_node(state: AgentState):
    """Kullanıcının niyetini analiz eder ve geçmişi göz önünde bulundurarak plan çıkarır."""
    system_prompt = build_planner_system_prompt(state.get('video_path', 'Belirtilmedi'))
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    
    response = llm.invoke(messages)
    return {"plan": response.content}


def executor_node(state: AgentState):
    """Planı ve (varsa) geri bildirimi alıp toolları kullanarak uygular."""
    system_prompt = build_executor_system_prompt(
        video_path=state.get('video_path', 'Belirtilmedi'),
        plan=state['plan'],
        feedback=state.get('feedback', '')
    )
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def reviewer_node(state: AgentState):
    """İşlemcinin sonuçlarını inceler. Yeterliyse bitirir, eksikse geri gönderir."""
    structured_llm = llm.with_structured_output(ReviewResult)

    system_prompt = build_reviewer_system_prompt(
        user_query=state['user_query'],
        plan=state['plan']
    )
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    result = structured_llm.invoke(messages)
    
    if result.is_complete:
        return {"final_answer": result.feedback_or_answer, "feedback": ""}
    else:
        return {"feedback": result.feedback_or_answer}

def route_after_executor(state: AgentState):
    """İşlemciden sonra tool çağrısı yapılıp yapılmayacağına karar verir."""
    last_message = state["messages"][-1]
    # Eğer İşlemci (LLM) bir tool çağırmaya karar verdiyse ToolNode'a git
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Tool çağrısı yoksa, kendi yorumunu yaptı demektir, Reviewer'a gönder.
    return "reviewer"

def route_after_reviewer(state: AgentState):
    """Denetleyicinin kararına göre sistemi bitirir veya İşlemciye geri döndürür."""
    if state.get("final_answer"):
        return END
    return "executor"

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

# İşlemci ya Tool çalıştıracak ya da Reviewer'a gidecek
workflow.add_conditional_edges("executor", route_after_executor)

# Tools çalıştıktan sonra sonuçlarla birlikte tekrar işlemciye dönmeli
workflow.add_edge("tools", "executor")

# Reviewer karar verecek: Ya bitecek (END) ya da İşlemciye (executor) geri dönecek
workflow.add_conditional_edges("reviewer", route_after_reviewer)

# Sistemi derle
video_agent_app = workflow.compile()