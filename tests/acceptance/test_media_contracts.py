from unittest.mock import patch
import numpy as np
from support import OfflineCase, FakeVideoReader
from utils import video_process as media
from utils.vlm import VLM_Manager
from utils import vlm as vlm_module


class MediaContracts(OfflineCase):
    """MAN-M01/M02/M03. No media files encoded, no VLM service calls."""
    def setUp(self):
        super().setUp()
        for name, value in {'VLM_MIN_FRAMES':8, 'VLM_MAX_FRAMES':128,
                            'VLM_MAX_EDGE':448, 'VLM_DIMENSION_ALIGNMENT':32,
                            'VLM_MIN_ENCODE_FPS':.25, 'VLM_MAX_ENCODE_FPS':30}.items():
            self.enterContext(patch.object(vlm_module, name, value))

    def test_sampling_respects_frame_budget_and_source_offset(self):
        with patch.object(media, 'VideoReader', FakeVideoReader):
            frames, start, end = media.generate_frames('fake.mp4', 2., 6., FPS=5, max_frames=12)
        self.assertEqual(len(frames), 12)
        self.assertAlmostEqual(start, 2.)
        self.assertAlmostEqual(end, 6.)

    def test_encode_fps_preserves_duration_inside_supported_limits(self):
        for frames, duration in ((100,20.), (112,22.4), (128,60.)):
            fps = VLM_Manager.calculate_encode_fps(frames, duration)
            self.assertAlmostEqual(frames / fps, duration)

    def test_preparation_minimum_frames_and_alignment(self):
        model = VLM_Manager.__new__(VLM_Manager)
        prepared = model._prepare_frames(np.zeros((160,90,3), dtype=np.uint8))
        self.assertGreaterEqual(prepared.shape[0], 8)
        self.assertEqual(prepared.shape[1] % 32, 0)
        self.assertEqual(prepared.shape[2] % 32, 0)
