import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from langchain_core.messages import HumanMessage
from support import OfflineCase, server_module, report


class BackendContracts(OfflineCase):
    """MAN-U01/U04/U05/R01. Real worker/events, no HTTP listener or native models."""
    def setUp(self):
        super().setUp()
        self.server = self.enterContext(server_module())
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.enterContext(patch.object(self.server, 'RUNS_DIR', self.root))
        self.enterContext(patch.object(self.server, 'REPORTS_DIR', self.root/'reports'))
        self.session = self.server._get_or_create_session('isolated')
        self.session.active_video_path = str(self.root/'video.mp4')

    def run_job(self, mode):
        job = self.server.JobState(job_id='test', session_id='isolated', mode=mode)
        self.assertTrue(self.session.operation_lock.acquire(blocking=False))
        self.server._run_job(job, 'test message')
        self.assertTrue(job.done_event.is_set())
        self.assertFalse(self.session.operation_lock.locked())
        return list(job.queue.queue)

    def test_report_state_does_not_reuse_chat_history(self):
        self.session.lc_messages = [HumanMessage(content='PRIVATE-CHAT')]
        state = self.server._build_initial_state('report', self.session)
        self.assertEqual(state['video_path'], self.session.active_video_path)
        self.assertNotIn('PRIVATE-CHAT', str(state['conversation_messages']))

    def test_approved_report_is_downloadable_and_does_not_mutate_chat(self):
        self.session.chat_history = [{'role':'user', 'content':'existing'}]
        self.server.video_agent_app.stream.return_value = iter([{'reviewer': {'report':report(), 'final_answer':json.dumps(report())}}])
        events = self.run_job('report')
        final = next(e for e in events if e['type']=='report_final')
        self.assertEqual(json.loads(Path(final['download_path']).read_text()), report())
        self.assertTrue(final['download_url'].startswith('/media/reports/'))
        self.assertEqual(self.session.chat_history, [{'role':'user','content':'existing'}])
        self.assertEqual([e['type'] for e in events].count('done'), 1)

    def test_failure_final_text_is_not_published_as_report(self):
        self.server.video_agent_app.stream.return_value = iter([{'reviewer': {'final_answer':'Analiz tamamlanmış sayılmamalıdır.'}}])
        events = self.run_job('report')
        self.assertIn('job_error', [e['type'] for e in events])
        self.assertNotIn('report_final', [e['type'] for e in events])
        self.assertEqual(list(self.root.glob('reports/*.json')), [])

    def test_other_session_cannot_reuse_model_context(self):
        other = self.server._get_or_create_session('other')
        self.assertIsNot(other.models, self.session.models)
        self.assertIsNot(other.lc_messages, self.session.lc_messages)
        self.assertIsNot(other.operation_lock, self.session.operation_lock)

    def test_pre_cancelled_job_never_calls_model_or_updates_history(self):
        job = self.server.JobState(job_id='cancel', session_id='isolated', mode='chat')
        job.cancel_event.set()
        self.session.operation_lock.acquire()
        self.server._run_job(job, 'ignored')
        self.server.video_agent_app.stream.assert_not_called()
        self.assertEqual(self.session.lc_messages, [])
        events = list(job.queue.queue)
        self.assertIn('job_cancelled', [e['type'] for e in events])
        self.assertEqual([e['type'] for e in events].count('done'), 1)
