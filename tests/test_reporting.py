"""Rapor regresyonları: gerçek API, video veya model ağırlıkları gerektirmez."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from utils.reporting import REPORT_TASK, validate_report


VIDEO_PATH = "/test/video.mp4"
REPORT = {
    "ozet": "Dört aday aralık incelendi; son kesitte fiziksel çatışma görülüyor.",
    "olaylar": [{"saniye": 62.2, "aciklama": "62.2–70.6 saniyelik kesitin başlangıcı; fiziksel çatışma."}],
    "risk_seviyesi": "yuksek",
    "eylemler": [],
}
REPORT_JSON = json.dumps(REPORT, ensure_ascii=False)
SEGMENTS = [(12.8, 19.8), (20.4, 27.8), (31.9, 51.0), (62.2, 70.7)]


def evidence():
    def message(name, call_id, data):
        return ToolMessage(
            name=name,
            tool_call_id=call_id,
            content=json.dumps({"ok": True, "data": {"video_path": VIDEO_PATH, **data}}),
        )

    messages = [message("run_abnormal_event_segmenter", "segment", {
        "analysis_scope": "full_video",
        "video": {"duration_sec": 70.6},
        "segments": [{"start_time": start, "end_time": end} for start, end in SEGMENTS],
    })]
    for index, (start, end) in enumerate(SEGMENTS):
        messages.append(message("analyze_video_with_vlm", f"visual_{index}", {
            "effective_range": {"start_sec": start, "end_sec": min(end, 70.6)},
            "vlm_response": "[video_url-mp4 n=35]\n" + (
                "Fiziksel çatışma görülüyor." if index == 3 else "Olağan faaliyetler görülüyor."
            ),
        }))
    return messages


class ReportValidationTests(unittest.TestCase):
    def test_supported_presentation_wrappers(self):
        variants = [
            REPORT_JSON,
            " \n" + REPORT_JSON + "\n ",
            "```json\n" + REPORT_JSON + "\n```",
            "```\n" + REPORT_JSON + "\n```",
            "json\n" + REPORT_JSON,
            "Detaylı analiz tamamlandı.\n\n```json\n" + REPORT_JSON + "\n```",
            "Rapor:\r\n```JSON\r\n" + REPORT_JSON + "\r\n```\r\nAnaliz tamamlandı.",
        ]
        for answer in variants:
            with self.subTest(answer=answer[:50]):
                self.assertEqual(validate_report(answer, evidence(), VIDEO_PATH), REPORT)

    def test_json_string_content_is_not_rewritten(self):
        report = copy.deepcopy(REPORT)
        report["ozet"] = 'Metinde {parantez}, "tırnak" ve ``` işareti var.\nİkinci satır.\u2028Unicode ayraç.'
        answer = "```json\n" + json.dumps(report, ensure_ascii=False) + "\n```"
        self.assertEqual(validate_report(answer, evidence(), VIDEO_PATH), report)

    def test_ambiguous_or_broken_json_is_not_salvaged(self):
        fenced = "```json\n" + REPORT_JSON + "\n```"
        variants = [
            "",
            "Rapor hazırlanamadı.",
            REPORT_JSON + REPORT_JSON,
            fenced + "\n" + fenced,
            fenced + '\n{"risk_seviyesi": "dusuk"}',
            "```json\n" + REPORT_JSON,
            "```python\n" + REPORT_JSON + "\n```",
            '```json\n{"ozet": "yarım kalan rapor"\n```',
            "```json\n[" + REPORT_JSON + "]\n```",
        ]
        for answer in variants:
            with self.subTest(answer=answer[:50]):
                with self.assertRaises(ValueError):
                    validate_report(answer, evidence(), VIDEO_PATH)

    def test_wrapper_does_not_bypass_report_schema(self):
        for changes in [
            {"eylemler": ["Bir eylem"]},
            {"risk_seviyesi": "bilinmiyor"},
            {"ozet": "   "},
            {"fazladan_alan": True},
            {"olaylar": [{"saniye": "62.2", "aciklama": "Olay"}]},
            {"olaylar": [{"saniye": float("nan"), "aciklama": "Olay"}]},
        ]:
            with self.subTest(changes=changes):
                answer = "```json\n" + json.dumps({**REPORT, **changes}) + "\n```"
                with self.assertRaises(ValueError):
                    validate_report(answer, evidence(), VIDEO_PATH)

    def test_wrapper_does_not_bypass_visual_coverage(self):
        answer = "```json\n" + REPORT_JSON + "\n```"
        for messages, target in [(evidence()[:-1], VIDEO_PATH), (evidence()[1:], VIDEO_PATH), (evidence(), "/other.mp4")]:
            with self.subTest(target=target, count=len(messages)):
                with self.assertRaises(ValueError):
                    validate_report(answer, messages, target)

    def test_wrapper_does_not_bypass_event_time_checks(self):
        for seconds in [-1, 60, 70.6, 90]:
            with self.subTest(seconds=seconds):
                report = {**REPORT, "olaylar": [{"saniye": seconds, "aciklama": "Olay"}]}
                with self.assertRaises(ValueError):
                    validate_report("```json\n" + json.dumps(report) + "\n```", evidence(), VIDEO_PATH)


class ReportGraphTests(unittest.TestCase):
    def setUp(self):
        # Graph ve düğümler gerçek koddan yüklenir; yalnız harici model/tool sınırları taklit edilir.
        spec = importlib.util.spec_from_file_location(
            "_report_test_agents", Path(__file__).resolve().parents[1] / "utils" / "agents.py"
        )
        self.agents = importlib.util.module_from_spec(spec)
        tool_module = types.ModuleType("utils.tools")
        tool_module.tools = []
        with patch.dict(sys.modules, {spec.name: self.agents, "utils.tools": tool_module}), patch("langchain_openai.ChatOpenAI"):
            spec.loader.exec_module(self.agents)
        self.agents.llm = MagicMock()
        self.agents.llm_with_tools = MagicMock()
        self.agents._tool_node = MagicMock()
        self.plan = self.agents.PlanResult(needs_tool=True, reasoning="Anomalileri incele.", steps=[])
        self.review = self.agents.ReviewResult(is_complete=True, feedback="Kanıtlar yeterli.")
        self.reviewer = MagicMock()
        self.reviewer.invoke.return_value = self.review
        self.agents.llm.with_structured_output.side_effect = lambda schema: (
            MagicMock(invoke=MagicMock(return_value=self.plan))
            if schema is self.agents.PlanResult else self.reviewer
        )
        self.state = {
            "output_mode": "report", "report": None, "user_query": REPORT_TASK,
            "video_path": VIDEO_PATH, "video_paths": [VIDEO_PATH], "image_paths": [],
            "conversation_messages": [HumanMessage(content=REPORT_TASK)],
            "messages": [], "plan": "", "feedback": "", "review_route": "",
            "final_answer": "", "tool_rounds": 0, "review_loops": 0,
        }

    def prepare_graph(self):
        self.agents.llm_with_tools.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "id": "segment", "name": "run_abnormal_event_segmenter", "args": {"video_path": VIDEO_PATH},
            }]),
            AIMessage(content="", tool_calls=[{
                "id": f"visual_{i}", "name": "analyze_video_with_vlm",
                "args": {"video_path": VIDEO_PATH, "query": "İncele", "start_sec": start, "end_sec": end},
            } for i, (start, end) in enumerate(SEGMENTS)]),
            AIMessage(content="Detaylı analiz tamamlandı.\n```json\n" + REPORT_JSON + "\n```"),
        ]
        messages = evidence()
        self.agents._tool_node.invoke.side_effect = [{"messages": messages[:1]}, {"messages": messages[1:]}]

    def assert_completed_report(self, state):
        self.assertEqual(state["report"], REPORT)
        self.assertEqual(json.loads(state["final_answer"]), REPORT)
        self.assertEqual(state["review_loops"], 0)
        self.assertEqual(self.agents._tool_node.invoke.call_count, 2)
        self.assertEqual(self.agents.llm_with_tools.invoke.call_count, 3)
        self.assertEqual(self.reviewer.invoke.call_count, 1)

    def test_gradio_sync_graph_finishes_without_reanalyzing(self):
        self.prepare_graph()
        final = self.agents.video_agent_app.invoke(self.state, {"recursion_limit": 40})
        self.assert_completed_report(final)

    def test_async_stream_returns_clean_json(self):
        self.prepare_graph()

        async def run():
            updates = {}
            async for data in self.agents.video_agent_app.astream(
                self.state, {"recursion_limit": 40},
            ):
                for update in data.values():
                    updates.update(update)
            return updates

        self.assert_completed_report(asyncio.run(run()))

    def test_reviewer_rejection_still_blocks_report(self):
        self.state["messages"] = evidence() + [AIMessage(content="```json\n" + REPORT_JSON + "\n```")]
        self.reviewer.invoke.return_value = self.agents.ReviewResult(is_complete=False, feedback="Kanıt yetersiz.")
        result = self.agents.reviewer_node(self.state)
        self.assertNotIn("report", result)
        self.assertEqual(result["final_answer"], "")

    def test_approval_cannot_bypass_missing_visual_evidence(self):
        self.state["messages"] = evidence()[:-1] + [AIMessage(content=REPORT_JSON)]
        result = self.agents.reviewer_node(self.state)
        self.assertNotIn("report", result)
        self.assertEqual(result["final_answer"], "")

    def test_approval_cannot_bypass_invalid_json(self):
        self.state["messages"] = evidence() + [AIMessage(content='```json\n{"ozet": "yarım"\n```')]
        self.state["review_loops"] = self.agents.MAX_REVIEW_LOOPS - 1
        result = self.agents.reviewer_node(self.state)
        self.assertNotIn("report", result)
        self.assertIn("doğrulanmış bir nihai yanıt hazırlayamadım", result["final_answer"])

    def test_chat_response_is_not_normalized(self):
        answer = "Merhaba.\n```json\n" + REPORT_JSON + "\n```"
        self.state.update(output_mode="chat", messages=[AIMessage(content=answer)])
        result = self.agents.reviewer_node(self.state)
        self.assertEqual(result["final_answer"], answer)
        self.assertNotIn("report", result)


if __name__ == "__main__":
    unittest.main()
