import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from support import OfflineCase, tools_module


class ToolBoundaries(OfflineCase):
    """MAN-M04/G02/G04: actual tool wrappers, fake metadata/models."""
    def setUp(self):
        super().setUp()
        self.tools = self.enterContext(tools_module())
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.video = Path(self.temp)/'video.mp4'
        self.video.write_bytes(b'FAKE-SOURCE')
        self.metadata = {'duration_sec':10.,'fps':30.,'frame_count':300,'width':90,'height':160}
        self.enterContext(patch.object(self.tools, '_get_video_metadata', return_value=self.metadata))

    def test_ranges_reject_negative_reversed_zero_and_nonfinite(self):
        for start, end in ((-1.,2.),(2.,2.),(3.,2.),(float('nan'),4.),(0.,float('inf'))):
            with self.subTest(start=start,end=end), self.assertRaises(self.tools.ToolInputError) as cm:
                self.tools._validate_video_range(str(self.video),start,end)
            self.assertEqual(cm.exception.code,'INVALID_TIME_RANGE')

    def test_only_end_is_clamped(self):
        start, end, _, clamped = self.tools._validate_video_range(str(self.video),9.,15.)
        self.assertEqual((start,end,clamped),(9.,10.,True))
        with self.assertRaises(self.tools.ToolInputError) as cm:
            self.tools._validate_video_range(str(self.video),10.,15.)
        self.assertEqual(cm.exception.code,'TIME_OUT_OF_RANGE')

    def test_empty_segment_result_contains_warning_not_normality_claim(self):
        model = MagicMock()
        model.analyze.return_value = []
        with patch.object(self.tools,'get_anomaly_segment_model',return_value=model):
            result = json.loads(self.tools.run_abnormal_event_segmenter.invoke({'video_path':str(self.video)}))
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['segments'],[])
        self.assertEqual(result['warnings'][0]['code'],'NO_SEGMENTS_ABOVE_THRESHOLD')
        self.assertEqual(model.analyze.call_args.kwargs['threshold'],.3)

    def test_segmenter_failure_is_not_success(self):
        with patch.object(self.tools,'get_anomaly_segment_model',side_effect=RuntimeError('test failure')):
            result = json.loads(self.tools.run_abnormal_event_segmenter.invoke({'video_path':str(self.video)}))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'],'SEGMENTER_ERROR')

    def test_missing_video_does_not_load_model(self):
        with patch.object(self.tools,'get_anomaly_segment_model') as model:
            result = json.loads(self.tools.run_abnormal_event_segmenter.invoke({'video_path':str(self.video)+'.missing'}))
        model.assert_not_called()
        self.assertEqual(result['error']['code'],'FILE_NOT_FOUND')
