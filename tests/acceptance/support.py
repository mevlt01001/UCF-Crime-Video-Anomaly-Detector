"""Offline fixtures. No real models, API credentials or application server.

Run using unittest discovery with -s tests/acceptance (puts this directory on sys.path).
Load actual source modules; only external boundaries are replaced. No AST extraction.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import socket
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

VIDEO = '/acceptance/video-a.mp4'
OTHER_VIDEO = '/acceptance/video-b.mp4'


def load_source(relative):
    name = 'utils._acceptance_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses need their module present while definitions are evaluated.
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


class OfflineCase(unittest.TestCase):
    def enterContext(self, context):
        # unittest.TestCase.enterContext is only built in on Python 3.11+.
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden in offline tests')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('Network forbidden in offline tests')))


def envelope(data=None, ok=True, code='TEST_ERROR'):
    return json.dumps({'ok': ok, 'data': data or {}, 'warnings': [],
                       'error': None if ok else {'code': code, 'message': 'injected failure'}})


def pair(name, args, data=None, *, ok=True, call_id=None, code='TEST_ERROR'):
    call_id = call_id or uuid.uuid4().hex
    return [AIMessage(content='', tool_calls=[{'id': call_id, 'name': name, 'args': args}]),
            ToolMessage(content=envelope(data, ok, code), name=name, tool_call_id=call_id)]


def evidence(segments=((2., 5.),), visuals=((2., 5.),), video=VIDEO, archives='auto'):
    messages = pair('run_abnormal_event_segmenter', {'video_path': video}, {
        'video_path': video, 'video': {'duration_sec': 12.}, 'analysis_scope': 'full_video',
        'segments': [{'start_time': a, 'end_time': b} for a, b in segments]})
    for a, b in visuals:
        messages += pair('analyze_video_with_vlm', {'video_path': video, 'start_sec': a, 'end_sec': b}, {
            'video_path': video, 'effective_range': {'start_sec': a, 'end_sec': b},
            'vlm_response': 'İncelenen kesitte araç görünür; olay türü belirsiz.'})
    if archives == 'auto':
        archive_specs = [('olay_1', segments[0][0], segments[0][1], 'belirsiz')] if segments else []
    else:
        archive_specs = list(archives)
    for spec in archive_specs:
        olay_id, a, b, category = spec
        messages += pair('archive_anomaly_clip', {
            'video_path': video, 'start_sec': a, 'end_sec': b,
            'category': category, 'explanation': 'Görsel kanıt yetersiz; olay türü belirsiz.',
            'incident_id': olay_id}, {
            'video_path': video, 'category': category, 'incident_id': olay_id,
            'saved_range': {'start_sec': a, 'end_sec': b},
            'output_path': f'/acceptance/archive/{olay_id}/{a}-{b}.mp4', 'cache_hit': False})
    return messages


def report(actions=(), events=()):
    if not events:
        events = [{'olay_id': 'olay_1', 'saniye': 2., 'aciklama': 'İncelenen kesitte olay türü belirsiz.'}]
    return {'ozet': 'Yalnız belirtilen aralık incelendi; kesin güvenlik garantisi yok.',
            'olaylar': list(events), 'risk_seviyesi': 'orta', 'eylemler': list(actions)}


def state(messages=(), mode='report'):
    from utils.reporting import REPORT_TASK
    return {'output_mode': mode, 'report': None, 'user_query': REPORT_TASK,
            'video_path': VIDEO, 'video_paths': [VIDEO], 'image_paths': [],
            'conversation_messages': [HumanMessage(content=REPORT_TASK)],
            'messages': list(messages), 'plan': '', 'feedback': '', 'review_route': '',
            'final_answer': '', 'tool_rounds': 0, 'review_loops': 0}


@contextmanager
def agent_module():
    boundary = types.ModuleType('utils.tools')
    boundary.tools = []
    boundary.chat_tools = []
    boundary.report_tools = []
    # Never instantiate an actual ChatOpenAI client or read local .env here.
    with patch.dict(sys.modules, {'utils.tools': boundary}), patch('dotenv.load_dotenv'), patch('langchain_openai.ChatOpenAI'):
        agent = load_source('utils/agents.py')
    yield agent


@contextmanager
def server_module():
    # ui_backend.__init__ normally imports server and thereby the live agent.
    package = types.ModuleType('ui_backend')
    package.__path__ = [str(ROOT / 'ui_backend')]
    agent = types.ModuleType('utils.agents')
    agent.video_agent_app = MagicMock()
    with patch.dict(sys.modules, {'ui_backend': package, 'utils.agents': agent}), patch.object(Path, 'mkdir'):
        server = load_source('ui_backend/server.py')
    yield server


class FakeVideoReader:
    """RGB portrait reader: dimensions reveal aspect-ratio distortion."""
    def __init__(self, path, ctx=None, width=0, height=0, **kwargs):
        self.width, self.height = width or 90, height or 160
    def __len__(self):
        return 300
    def get_avg_fps(self):
        return 30.
    def get_batch(self, indices):
        import numpy as np
        arr = np.zeros((len(indices), self.height, self.width, 3), dtype=np.uint8)
        return types.SimpleNamespace(asnumpy=lambda: arr)


@contextmanager
def tools_module():
    analyzer = types.ModuleType('utils.video_analyzer_model')
    analyzer.Video_Analyzer = MagicMock(name='Video_Analyzer')
    analyzer.pick_device = lambda: types.SimpleNamespace(type='cpu')
    with patch.dict(sys.modules, {'utils.video_analyzer_model': analyzer}), \
            patch('dotenv.load_dotenv'), patch('utils.env.load_env'), patch.dict('os.environ', {}, clear=True):
        tools = load_source('utils/tools.py')
    yield tools
