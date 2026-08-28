import json
import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from utils.prompts import build_executor_system_prompt, build_planner_system_prompt, build_reviewer_system_prompt
from utils.tools import chat_tools, report_tools
from utils.reporting import report_instructions, validate_report
from utils.action_records import action_instructions
from utils.usage_tracking import invoke_config
from utils.video_context import prepare_video_context, video_context_prompt

llm = ChatOpenAI(
    model=os.environ.get("EVREN_LLM_MODEL", "llm-fast"),
    api_key=os.environ.get("EVREN_API_KEY"),
    base_url=os.environ.get("EVREN_URL") or os.environ.get("EVREN_BASE_URL"),
    temperature=0.0,
)
_chat_tool_node = ToolNode(chat_tools)
_report_tool_node = ToolNode(report_tools)

# Report: segmenter(1) + VLM×N + archive×M; 4-seg barber ≈8, +failed/retry → 14
MAX_TOOL_ROUNDS = 14
MAX_REVIEW_LOOPS = 2


def _tools_for_mode(output_mode):
    return report_tools if output_mode == "report" else chat_tools


def _tool_catalog_for_mode(output_mode):
    return _build_tool_catalog(_tools_for_mode(output_mode))


def _build_tool_catalog(registered_tools) -> str:
    """Planner/Reviewer yetenek bilgisini kayıtlı tool şemalarından üretir."""
    catalog = []
    for registered_tool in registered_tools:
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
    is_complete: bool = Field(description="Son executor cevabı kullanıcıya sunulmaya uygun mu?")
    route_to: Literal["planner", "executor"] | None = None
    feedback: str = Field(description="Yalnız iç denetim gerekçesi veya düzeltme talebi; kullanıcı cevabı değildir.")


class AgentState(TypedDict):
    output_mode: Literal["chat", "report"]
    report: dict | None
    user_query: str
    video_path: str
    video_paths: list[str]
    image_paths: list[str]
    # Only this job's technical snapshot; never appended to conversation_messages.
    video_context: dict | None
    # Kalıcı, kullanıcıya dönük hafıza; tool/node mesajı içermez.
    conversation_messages: Sequence[BaseMessage]
    # Yalnızca bu görevin çalışma alanı; graph bitince dışarı taşınmaz.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: str
    feedback: str
    review_route: str
    final_answer: str
    answer_approved: bool
    tool_rounds: int
    review_loops: int


def _target_video(state: AgentState) -> str:
    return state.get("video_path") or (state.get("video_paths") or [None])[0] or "Belirtilmedi"


def _conversation(state: AgentState) -> list[BaseMessage]:
    return list(state.get("conversation_messages") or [])


def _work(state: AgentState) -> list[BaseMessage]:
    return list(state.get("messages") or [])


def _video_path_for_context(state: AgentState):
    return state.get("video_path") or (state.get("video_paths") or [None])[0] or None


def planner_node(state: AgentState):
    """Temiz sohbet context'iyle plan üretir; tool izi kalıcı hafızaya sızmaz."""
    video_path = _video_path_for_context(state)
    context = prepare_video_context(video_path)
    prompt = build_planner_system_prompt(
        _target_video(state),
        tool_catalog=_tool_catalog_for_mode(state.get("output_mode")),
        previous_plan=state.get("plan", ""),
        feedback=state.get("feedback", ""),
    )
    prompt += video_context_prompt(context, video_path)
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    if state.get("output_mode") == "report":
        messages[0] = SystemMessage(content=prompt + report_instructions())
    result = llm.with_structured_output(PlanResult).invoke(messages, invoke_config())
    return {
        "plan": json.dumps(result.model_dump(), ensure_ascii=False),
        "feedback": "",
        "review_route": "",
        "video_context": context,
    }


def executor_node(state: AgentState):
    """Planı temiz sohbet hafızası ve sadece mevcut görevin çalışma iziyle uygular."""
    prompt = build_executor_system_prompt(
        video_path=_target_video(state),
        plan=state.get("plan", ""),
        feedback=state.get("feedback", ""),
    )
    prompt += video_context_prompt(state.get("video_context"), _video_path_for_context(state))
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    if state.get("output_mode") == "report":
        messages[0] = SystemMessage(content=prompt + "\nBu modda doğal dil nihai cevap yerine JSON raporu üret.\n"
                                    + report_instructions() + action_instructions(_work(state), _target_video(state)))
    bound = llm.bind_tools(_tools_for_mode(state.get("output_mode")))
    return {"messages": [bound.invoke(messages, invoke_config())], "feedback": ""}


def _executor_answer(state: AgentState) -> str:
    """Yalnız son, tool çağrısı içermeyen executor mesajı cevap adayıdır."""
    messages = _work(state)
    if not messages:
        return ""
    message = messages[-1]
    if not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls:
        return ""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        block if isinstance(block, str) else block["text"]
        for block in content
        if isinstance(block, str)
        or (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
    ).strip()


def tools_node(state: AgentState):
    node = _report_tool_node if state.get("output_mode") == "report" else _chat_tool_node
    out = node.invoke({"messages": _work(state)})
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
        tool_catalog=_tool_catalog_for_mode(state.get("output_mode")),
    )
    prompt += video_context_prompt(state.get("video_context"), _video_path_for_context(state))
    report = None
    validation_error = ""
    answer = _executor_answer(state)
    if state.get("output_mode") == "report":
        prompt += report_instructions() + "\nJSON raporunu, risk gerekçesini ve kanıt yeterliliğini denetle. Eksik/yetersiz analizi onaylama."
        try:
            report = validate_report(answer, _work(state), _target_video(state))
            # Ham taslağı değil, doğrulanmış nesnenin JSON'unu ver.
            answer = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            prompt += (
                "\nKodun şema, zaman ve kapsam doğrulaması başarılı. JSON dışındaki sunum "
                "metni/kod çiti temizlendi; yalnız bu biçim farkı için yeniden analiz isteme. "
                "Ham taslak ve tool kanıtları aşağıdaki mesajlarda korunuyor; içerik ve "
                "risk gerekçesini denetlemeye devam et. Onaylanırsa kullanıcıya yalnız "
                "şu JSON sunulacak:\n" + answer
            )
        except (ValueError, TypeError, KeyError) as exc:
            validation_error = str(exc)
            prompt += "\nKod doğrulaması başarısız: " + validation_error
    messages = [SystemMessage(content=prompt)] + _conversation(state) + _work(state)
    result = llm.with_structured_output(ReviewResult).invoke(messages, invoke_config())
    loops = int(state.get("review_loops") or 0)

    feedback = result.feedback.strip()
    if validation_error:
        feedback = "Rapor doğrulama hatası: " + validation_error + "\n" + feedback
    if result.is_complete and answer and not validation_error:
        output = {"final_answer": answer, "answer_approved": True, "feedback": feedback, "review_route": "", "review_loops": loops}
        if state.get("output_mode") == "report":
            output["report"] = report
        return output

    # Onaylanmamış taslağı, tool verisini veya denetim notunu kullanıcıya sızdırma.
    if loops + 1 >= MAX_REVIEW_LOOPS:
        return {
            "final_answer": "İsteğiniz için doğrulanmış bir nihai yanıt hazırlayamadım. İşlem deneme sınırına ulaştığı için durduruldu; analiz tamamlanmış sayılmamalıdır.",
            "answer_approved": False,
            "feedback": feedback,
            "review_route": "",
            "review_loops": loops + 1,
        }

    return {
        "feedback": feedback or "Kullanıcıya yönelik, eldeki kanıtlarla desteklenen bir cevap hazırla.",
        "review_route": "executor" if result.is_complete else (result.route_to or "executor"),
        "final_answer": "",
        "answer_approved": False,
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
