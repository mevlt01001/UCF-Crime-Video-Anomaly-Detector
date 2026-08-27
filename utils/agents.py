import json
import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from utils.prompts import build_executor_system_prompt, build_planner_system_prompt, build_reviewer_system_prompt
from utils.tools import tools

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("EVREN_LLM_MODEL", "llm-fast"),
    api_key=os.environ.get("EVREN_API_KEY"),
    base_url=os.environ.get("EVREN_URL") or os.environ.get("EVREN_BASE_URL"),
    temperature=0.0,
)
llm_with_tools = llm.bind_tools(tools)
_tool_node = ToolNode(tools)

MAX_TOOL_ROUNDS = 8
MAX_REVIEW_LOOPS = 2


def _build_tool_catalog() -> str:
    """Planner/Reviewer yetenek bilgisini kayıtlı tool şemalarından üretir."""
    catalog = []
    for registered_tool in tools:
        schema_model = registered_tool.args_schema
        if hasattr(schema_model, "model_json_schema"):
            parameters = schema_model.model_json_schema()
        else:
            parameters = schema_model.schema()
        catalog.append(
            {
                "name": registered_tool.name,
                "description": registered_tool.description,
                "parameters": parameters,
            }
        )
    return json.dumps(catalog, ensure_ascii=False)


TOOL_CATALOG = _build_tool_catalog()


class PlanStep(BaseModel):
    step: int
    tool: str
    goal: str


class PlanResult(BaseModel):
    needs_tool: bool
    reasoning: str
    steps: list[PlanStep] = Field(default_factory=list)
    direct_answer: str | None = None


class ReviewResult(BaseModel):
    is_complete: bool
    route_to: Literal["planner", "executor"] | None = None
    feedback_or_answer: str


class AgentState(TypedDict):
    user_query: str
    video_path: str
    video_paths: list[str]
    image_paths: list[str]
    # Kalıcı, kullanıcıya dönük hafıza; tool/node mesajı içermez.
    conversation_messages: Sequence[BaseMessage]
    # Yalnızca bu görevin çalışma alanı; graph bitince dışarı taşınmaz.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: str
    feedback: str
    review_route: str
    final_answer: str
    tool_rounds: int
    review_loops: int


def _target_video(state: AgentState) -> str:
    return state.get("video_path") or (state.get("video_paths") or [None])[0] or "Belirtilmedi"


def _conversation(state: AgentState) -> list[BaseMessage]:
    return list(state.get("conversation_messages") or [])


def _work(state: AgentState) -> list[BaseMessage]:
    return list(state.get("messages") or [])


def planner_node(state: AgentState):
    """Temiz sohbet context'iyle plan üretir; tool izi kalıcı hafızaya sızmaz."""
    prompt = build_planner_system_prompt(
        _target_video(state),
        tool_catalog=TOOL_CATALOG,
        previous_plan=state.get("plan", ""),
        feedback=state.get("feedback", ""),
    )
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    result = llm.with_structured_output(PlanResult).invoke(messages)
    return {
        "plan": json.dumps(result.model_dump(), ensure_ascii=False),
        "feedback": "",
        "review_route": "",
    }


def executor_node(state: AgentState):
    """Planı temiz sohbet hafızası ve sadece mevcut görevin çalışma iziyle uygular."""
    prompt = build_executor_system_prompt(
        video_path=_target_video(state),
        plan=state.get("plan", ""),
        feedback=state.get("feedback", ""),
    )
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    return {"messages": [llm_with_tools.invoke(messages)], "feedback": ""}


def _last_usable_text(state: AgentState) -> str:
    for msg in reversed(_work(state)):
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str) and content.strip():
            return content.strip()
    return "İşlem tamamlanamadı; kullanılabilir bir sonuç üretilemedi."


def tools_node(state: AgentState):
    out = _tool_node.invoke({"messages": _work(state)})
    out["tool_rounds"] = int(state.get("tool_rounds") or 0) + 1
    return out


def tool_limit_node(state: AgentState):
    """Tool protokolünü bozmadan bekleyen çağrıları kapatır ve reviewer'a geçirir."""
    last = _work(state)[-1]
    messages = [
        ToolMessage(
            content=f"Araç turu güvenlik sınırına ulaştı ({MAX_TOOL_ROUNDS}); çağrı çalıştırılmadı.",
            tool_call_id=call["id"],
        )
        for call in (getattr(last, "tool_calls", None) or [])
    ]
    return {"messages": messages}


def reviewer_node(state: AgentState):
    prompt = build_reviewer_system_prompt(
        state["user_query"],
        state.get("plan", ""),
        tool_catalog=TOOL_CATALOG,
    )
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    result = llm.with_structured_output(ReviewResult).invoke(messages)
    loops = int(state.get("review_loops") or 0)

    if result.is_complete:
        answer = (result.feedback_or_answer or "").strip() or _last_usable_text(state)
        return {"final_answer": answer, "feedback": "", "review_route": "", "review_loops": loops}

    # Son düzeltme hakkından sonra graph cevapsız bitmesin. Elde edilen en iyi
    # executor çıktısını döndür; sonsuz reviewer döngüsüne girme.
    if loops + 1 >= MAX_REVIEW_LOOPS:
        return {
            "final_answer": _last_usable_text(state),
            "feedback": "",
            "review_route": "",
            "review_loops": loops + 1,
        }

    return {
        "feedback": result.feedback_or_answer,
        "review_route": result.route_to or "executor",
        "final_answer": "",
        "review_loops": loops + 1,
    }


def route_after_executor(state: AgentState):
    last = _work(state)[-1]
    if getattr(last, "tool_calls", None):
        if int(state.get("tool_rounds") or 0) < MAX_TOOL_ROUNDS:
            return "tools"
        return "tool_limit"
    return "reviewer"


def route_after_reviewer(state: AgentState):
    if state.get("final_answer") or int(state.get("review_loops") or 0) >= MAX_REVIEW_LOOPS:
        return END
    return "planner" if state.get("review_route") == "planner" else "executor"


workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", tools_node)
workflow.add_node("tool_limit", tool_limit_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", route_after_executor)
workflow.add_edge("tools", "executor")
workflow.add_edge("tool_limit", "reviewer")
workflow.add_conditional_edges("reviewer", route_after_reviewer)

video_agent_app = workflow.compile()
