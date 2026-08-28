import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage
from support import OfflineCase, agent_module, state, evidence, report, VIDEO
from utils.action_records import action_records


class AgentLifecycle(OfflineCase):
    """MAN-G03/G04/G05/G06: real nodes, offline model decisions."""
    def setUp(self):
        super().setUp()
        self.agent = self.enterContext(agent_module())
        self.agent.llm = MagicMock()
        self.executor_llm = MagicMock()
        self.agent.llm.bind_tools.return_value = self.executor_llm

    def test_executor_has_target_video_in_model_context(self):
        self.executor_llm.invoke.return_value = AIMessage(content='draft')
        self.agent.executor_node(state())
        sent = self.executor_llm.invoke.call_args.args[0]
        self.assertIn(VIDEO, sent[0].content)

    def test_chat_executor_does_not_bind_archive_tool(self):
        chat_tool = MagicMock()
        chat_tool.name = 'get_video_info'
        archive_tool = MagicMock()
        archive_tool.name = 'archive_anomaly_clip'
        self.agent.chat_tools = [chat_tool]
        self.agent.report_tools = [chat_tool, archive_tool]
        self.executor_llm.invoke.return_value = AIMessage(content='draft')
        self.agent.executor_node(state(mode='chat'))
        self.agent.llm.bind_tools.assert_called_once_with(self.agent.chat_tools)
        bound_tools = [tool.name for tool in self.agent.llm.bind_tools.call_args.args[0]]
        self.assertIn('get_video_info', bound_tools)
        self.assertNotIn('archive_anomaly_clip', bound_tools)

    def test_unexecuted_analysis_cannot_be_approved_as_report(self):
        self.agent.llm.with_structured_output.return_value.invoke.return_value = self.agent.ReviewResult(is_complete=True, feedback='Approve')
        output = self.agent.reviewer_node(state([AIMessage(content=json.dumps(report()))]))
        self.assertNotIn('report', output)
        self.assertEqual(output['final_answer'], '')
        self.assertIn('segmentasyon', output['feedback'])

    def test_exhausted_review_budget_returns_failure_not_report(self):
        self.agent.llm.with_structured_output.return_value.invoke.return_value = self.agent.ReviewResult(is_complete=False, feedback='Missing evidence')
        current = state([AIMessage(content=json.dumps(report()))])
        current['review_loops'] = self.agent.MAX_REVIEW_LOOPS - 1
        output = self.agent.reviewer_node(current)
        self.assertNotIn('report', output)
        self.assertIn('tamamlanmış sayılmamalıdır', output['final_answer'])

    def test_complete_report_is_canonical_json(self):
        self.agent.llm.with_structured_output.return_value.invoke.return_value = self.agent.ReviewResult(is_complete=True, feedback='Evidence sufficient')
        messages = evidence()
        records = action_records(messages, VIDEO)
        output = self.agent.reviewer_node(state(messages+[AIMessage(content='```json\n'+json.dumps(report(actions=records))+'\n```')]))
        self.assertEqual(json.loads(output['final_answer']), report(actions=records))
        self.assertEqual(output['report'], report(actions=records))

    def test_tool_budget_closes_pending_calls_without_executing(self):
        current = state([AIMessage(content='', tool_calls=[{'id':'pending','name':'archive_anomaly_clip','args':{}}])])
        current['tool_rounds'] = self.agent.MAX_TOOL_ROUNDS
        self.assertEqual(self.agent.route_after_executor(current), 'tool_limit')
        output = self.agent.tool_limit_node(current)
        self.assertEqual(len(output['messages']), 1)
        self.assertIsInstance(output['messages'][0], ToolMessage)
        self.assertEqual(output['messages'][0].tool_call_id, 'pending')
        self.assertIn('çağrı çalıştırılmadı', output['messages'][0].content)
