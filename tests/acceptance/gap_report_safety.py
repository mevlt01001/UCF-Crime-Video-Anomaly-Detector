"""Intentionally red acceptance requirements, NOT part of test_*.py discovery.

Run explicitly after deciding to fix the linked gaps. No expectedFailure/xfail:
a failure must remain visible, and a later pass must be a genuine improvement.
"""
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from support import OfflineCase, VIDEO, evidence, report, state, agent_module, server_module, FakeVideoReader
from utils.reporting import validate_report
from utils import video_process as media


class KnownReportGaps(OfflineCase):
    def test_KG01_reviewer_receives_uploaded_target_even_without_tool_results(self):
        # MAN-G03: planner/executor see a path but reviewer currently does not.
        with agent_module() as agent:
            agent.llm = MagicMock()
            agent.llm.with_structured_output.return_value.invoke.return_value = agent.ReviewResult(is_complete=False, feedback='retry')
            agent.reviewer_node(state([AIMessage(content=json.dumps(report()))]))
            sent = agent.llm.with_structured_output.return_value.invoke.call_args.args[0]
            self.assertTrue(any(VIDEO in str(message.content) for message in sent), 'Reviewer must know the actual target video.')

    def test_KG02_empty_segments_require_visual_review_before_normal_report(self):
        # MAN-G02: product requirement, not currently enforced by validate_report.
        normal_report = report()
        normal_report['olaylar'] = []
        normal_report['risk_seviyesi'] = 'dusuk'
        payload = json.dumps(normal_report)
        # Establish that the same report is valid when visual evidence exists.
        validate_report(payload, evidence(segments=(), visuals=((0., 12.),), archives=()), VIDEO)
        with self.assertRaises(ValueError):
            validate_report(payload, evidence(segments=(), visuals=(), archives=()), VIDEO)

    def test_KG03_portrait_video_is_not_stretched_to_landscape(self):
        # MAN-G01: generate_frames currently forces 448x336.
        with patch.object(media, 'VideoReader', FakeVideoReader):
            frames, _, _ = media.generate_frames('fake.mp4', 0., 1., FPS=5, max_frames=5)
        actual_ratio = frames.shape[2]/frames.shape[1]
        self.assertAlmostEqual(actual_ratio, 90/160, delta=.04, msg='Preserve content geometry before VLM encoding.')

    def test_KG04_failure_is_not_labelled_final_answer_approved(self):
        # MAN-G05: nonempty failure text is currently mistaken for approval.
        with server_module() as server:
            summary, _ = server._normalize_node_update('reviewer', {
                'final_answer':'İşlem deneme sınırına ulaştı; analiz tamamlanmış sayılmamalıdır.',
                'feedback':'Rapor doğrulama hatası', 'review_loops':2})
        self.assertNotIn('approved', summary)

    def test_KG05_unpaired_tool_result_cannot_count_as_visual_evidence(self):
        # MAN-R05: actions are paired strictly, visual/segment evidence currently is not.
        visual_report = report()
        visual_report['olaylar'] = []
        payload = json.dumps(visual_report)
        paired_results = evidence(archives=())
        # No archive/event requirement may mask missing call/result pairing.
        validate_report(payload, paired_results, VIDEO)
        orphan_results = [m for m in paired_results if m.type == 'tool']
        with self.assertRaises(ValueError):
            validate_report(payload, orphan_results, VIDEO)

    def test_KG06_segmenter_requires_checkpoint_before_loading_model(self):
        # MAN-S02: API Analyzer requires a checkpoint; agent tool may fall through with None.
        from support import tools_module
        with tools_module() as tools, patch.object(tools, '_resolve_checkpoint', return_value=None):
            with self.assertRaises(FileNotFoundError):
                tools.get_anomaly_segment_model()
