"""API regressions: real routes, fake external model/graph boundaries."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_agents = types.ModuleType('utils.agents')
        fake_agents.video_agent_app = MagicMock()
        with patch.dict(sys.modules, {'utils.agents': fake_agents}):
            cls.server = importlib.import_module('ui_backend.server')
            cls.runs = importlib.import_module('ui_backend.lab_runs')

    def setUp(self):
        self.server._sessions.clear()
        self.server._jobs.clear()
        self.client = TestClient(self.server.create_app(), raise_server_exceptions=False)
        self.session = self.server._get_or_create_session('a')

    def start(self, mode='chat'):
        path = '/api/jobs/analyzer' if mode == 'analyzer' else '/api/' + mode
        return self.client.post(path, json={'session_id': 'a', 'message': 'Merhaba'})

    def events(self, job):
        self.assertTrue(job.done_event.wait(3))
        return list(job.queue.queue)

    def test_chat_without_video(self):
        with patch.object(self.server.video_agent_app, 'stream', return_value=iter([
            {'reviewer': {'final_answer': 'Merhaba'}}
        ])):
            response = self.start()
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()['detail'], 'Önce video yükleyin.')
            self.assertFalse(self.server._jobs)
            self.assertEqual(self.session.lc_messages, [])
            self.assertFalse(self.session.operation_lock.locked())

    def test_analyzer_emits_zero_usage(self):
        self.session.active_video_path = '/tmp/example.mp4'
        with patch.object(self.server, 'run_analyzer', return_value=('segments', None)):
            job = self.server._jobs[self.start('analyzer').json()['job_id']]
            usage_events = [x for x in self.events(job) if x['type'] == 'usage_update']
            self.assertTrue(usage_events)
            final = usage_events[-1]
            self.assertEqual(final['total_tokens'], 0)
            self.assertIsNone(final['tokens_per_sec'])
            self.assertTrue(final['complete'])

    def test_analyzer_cancel_and_busy_session(self):
        self.session.active_video_path = '/tmp/example.mp4'
        entered, release = Event(), Event()
        def analyze(_):
            entered.set()
            release.wait(3)
            return 'segments', None
        with patch.object(self.server, 'run_analyzer', side_effect=analyze):
            response = self.start('analyzer')
            self.assertEqual(response.status_code, 200)
            job = self.server._jobs[response.json()['job_id']]
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.start().status_code, 409)
                self.assertEqual(self.client.post('/api/sessions/a/clear').status_code, 409)
                self.assertEqual(self.client.post('/api/analyzer', data={'session_id':'a'}).status_code, 409)
                self.assertEqual(self.client.post('/api/jobs/'+job.job_id+'/cancel', json={'session_id':'a'}).status_code, 200)
            finally:
                release.set()
            kinds = [x['type'] for x in self.events(job)]
            self.assertIn('job_cancelled', kinds)
            self.assertNotIn('analyzer_final', kinds)
            self.assertEqual(kinds.count('done'), 1)
            self.assertFalse(self.session.operation_lock.locked())

    def test_analyzer_error_is_not_success(self):
        self.session.active_video_path = '/tmp/example.mp4'
        with patch.object(self.server, 'run_analyzer', side_effect=RuntimeError('model failed')):
            response = self.client.post('/api/analyzer', data={'session_id':'a'})
            self.assertEqual(response.status_code, 500)
            self.assertIn('model failed', response.json()['detail'])
            job = self.server._jobs[self.start('analyzer').json()['job_id']]
            kinds = [x['type'] for x in self.events(job)]
            self.assertIn('job_error', kinds)
            self.assertNotIn('analyzer_final', kinds)
        self.assertFalse(self.session.operation_lock.locked())

    def test_models_are_session_local_and_clear_resets_them(self):
        class Manager:
            def __init__(self): self.history = []
            def clear_history(self): self.history = []
            def run(self, prompt, **kwargs):
                self.history.append(prompt)
                return '|'.join(self.history)
        module = types.ModuleType('utils.llm')
        module.LLM_Manager = Manager
        with patch.dict(sys.modules, {'utils.llm': module}):
            def ask(sid, text):
                return self.client.post('/api/llm', json={'session_id':sid, 'prompt':text}).json()['output']
            self.assertIn('secret-a', ask('a', 'secret-a'))
            self.assertNotIn('secret-a', ask('b', 'hello-b'))
            self.assertIn('secret-a', ask('a', 'followup'))
            self.assertEqual(self.client.post('/api/sessions/a/clear').status_code, 200)
            self.assertNotIn('secret-a', ask('a', 'fresh'))

    def test_vlm_context_isolation_and_clear(self):
        class Manager:
            def __init__(self): self.history = []
            def reset_context(self): self.history = []
            def run(self, prompt, **kwargs):
                self.history.append(prompt)
                return '|'.join(self.history)
        module = types.ModuleType('utils.vlm')
        module.VLM_Manager = Manager
        self.session.active_video_path = '/tmp/a.mp4'
        self.server._get_or_create_session('b').active_video_path = '/tmp/b.mp4'
        with patch.dict(sys.modules, {'utils.vlm': module}), patch.object(self.runs, '_frames_from_upload', return_value=(None, None)):
            def ask(sid, prompt, keep=True):
                response = self.client.post('/api/vlm', json={'session_id':sid, 'prompt':prompt, 'keep_history':keep})
                self.assertEqual(response.status_code, 200)
                return response.json()['output']
            ask('a', 'private-a')
            self.assertNotIn('private-a', ask('b', 'private-b'))
            self.assertNotIn('private-a', ask('a', 'fresh-a', False))
            self.assertIn('private-b', ask('b', 'followup-b'))

    def test_cancelled_chat_does_not_commit_history(self):
        self.session.active_video_path = '/tmp/example.mp4'
        entered, release = Event(), Event()
        def stream(*args):
            entered.set()
            release.wait(3)
            yield {'reviewer': {'final_answer': 'late answer'}}
        with patch.object(self.server.video_agent_app, 'stream', side_effect=stream):
            job = self.server._jobs[self.start().json()['job_id']]
            try:
                self.assertTrue(entered.wait(2))
                job.cancel_event.set()
            finally:
                release.set()
            kinds = [x['type'] for x in self.events(job)]
            self.assertNotIn('chat_final', kinds)
            self.assertEqual(self.session.lc_messages, [])
            self.assertEqual(self.session.chat_history, [])

    def test_manager_error_mode_preserves_gradio_default(self):
        from utils.llm import LLM_Manager
        manager = LLM_Manager.__new__(LLM_Manager)
        manager.history = []
        manager.llm = MagicMock()
        manager.llm.invoke.side_effect = RuntimeError('offline')
        self.assertIn('[LLM HATA]', manager.run('hello'))
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            manager.run('hello', raise_on_error=True)
        self.assertEqual(manager.history, [])
        self.session.models.llm = manager
        self.assertEqual(self.client.post('/api/llm', json={'session_id':'a', 'prompt':'hello'}).status_code, 500)

    def test_video_replacement_requires_new_session(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.server, 'UPLOADS_DIR', Path(directory)):
            def upload(sid):
                return self.client.post('/api/videos', data={'session_id':sid}, files={'file':('a.mp4', b'fake video', 'video/mp4')})
            self.assertEqual(upload('a').status_code, 200)
            old_path = self.session.active_video_path
            self.assertEqual(upload('a').status_code, 409)
            self.assertEqual(self.session.active_video_path, old_path)
            self.assertEqual(upload('b').status_code, 200)

    def test_lab_errors_propagate_and_checkpoint_required(self):
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            self.runs._timed(lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        # No heavyweight model initialization for missing checkpoints.
        with patch.object(self.runs, '_resolve_analyzer_checkpoint', return_value=None):
            with self.assertRaises(FileNotFoundError):
                self.runs.run_analyzer('/tmp/example.mp4')


if __name__ == '__main__':
    unittest.main()
