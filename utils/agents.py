import os

from typing import TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from utils.tools import tools
from utils.prompts import (
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
_tool_node = ToolNode(tools)

MAX_TOOL_ROUNDS = 8
MAX_REVIEW_LOOPS = 2


class AgentState(TypedDict):
    user_query: str
    video_path: str
    video_paths: list[str]
    image_paths: list[str]
    plan: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    feedback: str
    final_answer: str
    tool_rounds: int
    review_loops: int

class ReviewResult(BaseModel):
    is_complete: bool = Field(description="İşlemcinin bulguları kullanıcının isteğini tam olarak karşılıyor mu?")
    feedback_or_answer: str = Field(description="Eğer eksik varsa İşlemciye verilecek geri bildirim. Tamamsa kullanıcıya sunulacak nihai, derlenmiş cevap.")

def _target_video(state: AgentState) -> str:
    return state.get("video_path") or (state.get("video_paths") or [None])[0] or "Belirtilmedi"


def planner_node(state: AgentState):
    """Kullanıcının niyetini analiz eder ve geçmişi göz önünde bulundurarak plan çıkarır."""
    system_prompt = build_planner_system_prompt(_target_video(state))
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    
    response = llm.invoke(messages)
    return {"plan": response.content}


def executor_node(state: AgentState):
    """Planı ve (varsa) geri bildirimi alıp toolları kullanarak uygular."""
    system_prompt = build_executor_system_prompt(
        video_path=_target_video(state),
        plan=state['plan'],
        feedback=state.get('feedback', '')
    )
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _last_usable_text(state: AgentState) -> str:
    for msg in reversed(list(state.get("messages") or [])):
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str) and content.strip():
            return content.strip()
    return "İşlem durdu; derlenmiş cevap üretilemedi."


def tools_node(state: AgentState):
    out = _tool_node.invoke(state)
    out["tool_rounds"] = int(state.get("tool_rounds") or 0) + 1
    return out


def reviewer_node(state: AgentState):
    """İşlemcinin sonuçlarını inceler. Yeterliyse bitirir, eksikse geri gönderir."""
    structured_llm = llm.with_structured_output(ReviewResult)

    system_prompt = build_reviewer_system_prompt(
        user_query=state['user_query'],
        plan=state['plan']
    )
    
    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    result = structured_llm.invoke(messages)
    loops = int(state.get("review_loops") or 0)

    if result.is_complete or loops >= MAX_REVIEW_LOOPS:
        answer = (result.feedback_or_answer or "").strip() or _last_usable_text(state)
        return {"final_answer": answer, "feedback": "", "review_loops": loops}
    return {
        "feedback": result.feedback_or_answer,
        "final_answer": "",
        "review_loops": loops + 1,
    }

def route_after_executor(state: AgentState):
    """İşlemciden sonra tool çağrısı yapılıp yapılmayacağına karar verir."""
    last_message = state["messages"][-1]
    rounds = int(state.get("tool_rounds") or 0)
    if (
        hasattr(last_message, "tool_calls")
        and last_message.tool_calls
        and rounds < MAX_TOOL_ROUNDS
    ):
        return "tools"
    return "reviewer"

def route_after_reviewer(state: AgentState):
    """Denetleyicinin kararına göre sistemi bitirir veya İşlemciye geri döndürür."""
    if state.get("final_answer"):
        return END
    if int(state.get("review_loops") or 0) >= MAX_REVIEW_LOOPS:
        return END
    return "executor"

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", tools_node)
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