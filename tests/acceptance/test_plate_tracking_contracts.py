import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from support import OfflineCase
from utils import object_tracking as tracking, plate_detection as detection, plate_ocr as ocr


class TrackingContracts(OfflineCase):
    """MAN-O01/O02/O03/O04: no YOLO or ByteTrack construction."""
    def test_class_filters_are_normalized_and_unsupported_classes_rejected(self):
        names = {0: 'person', 2: 'car'}
        self.assertIsNone(tracking._class_ids(None, names))
        self.assertEqual(tracking._class_ids([' CAR ', 'car', 'person'], names), [0, 2])
        for classes in ([], ['gun'], ['plate']):
            with self.subTest(classes=classes), self.assertRaises(tracking.TrackingError) as cm:
                tracking._class_ids(classes, names)
            self.assertEqual(cm.exception.code, 'UNSUPPORTED_OBJECT_CLASS')

    def test_source_time_lower_bound(self):
        class Reader:
            def __len__(self): return 4
            def get_frame_timestamp(self, i): return [(0., .1), (.1, .35), (.35, .4), (.4, .8)][i]
        for second, expected in ((0., 0), (.1, 1), (.2, 2), (.4, 3), (.8, 4)):
            self.assertEqual(tracking._lower_frame(Reader(), second), expected)
            self.assertEqual(detection._lower_frame(Reader(), second), expected)

    def test_disappearance_splits_intervals(self):
        out = io.StringIO()
        ranges = tracking._Intervals(out)
        visible = {('class','car'): {'kind':'class', 'class':'car', 'track_id':None}}
        ranges.update(visible, 1., 2.)
        ranges.update({}, 2., 3.)
        ranges.update(visible, 3., 4.)
        ranges.finish()
        result = json.loads('[' + out.getvalue() + ']')
        self.assertEqual([(x['start_sec'], x['end_sec']) for x in result], [(1., 2.), (3., 4.)])

    def test_class_ranges_take_preview_priority_over_fragmented_tracks(self):
        ranges = tracking._Intervals(io.StringIO())
        for i in range(110):
            ranges.update({i: {'kind':'track', 'class':'car', 'track_id':i}}, i, i + 1)
        ranges.update({'car': {'kind':'class', 'class':'car', 'track_id':None}}, 111., 112.)
        ranges.finish()
        self.assertEqual(ranges.count, 111)
        self.assertEqual(len(ranges.preview), 100)
        self.assertEqual(ranges.preview[0]['kind'], 'class')

    def test_cache_tampering_invalidates_record(self):
        with tempfile.TemporaryDirectory() as td, patch.object(tracking, 'OUTPUT_ROOT', Path(td)):
            data, index = Path(td)/'frames.json', Path(td)/'index.json'
            data.write_bytes(b'[]')
            record = {'files':[{'path':str(data), 'size':2, 'sha256':hashlib.sha256(b'[]').hexdigest()}],
                      'result':{'data':{'processed_frame_count':0}, 'warnings':[]}}
            index.write_text(json.dumps(record))
            self.assertIsNotNone(tracking._cached(index))
            data.write_bytes(b'{}')  # Same size: digest must be checked.
            self.assertIsNone(tracking._cached(index))


class PlateContracts(OfflineCase):
    """MAN-P01/P02, MAN-C01/C02/C03/C04. Synthetic model outputs, not accuracy tests."""
    def test_letterbox_coordinates_map_to_original_pixels(self):
        class Model:
            def get_inputs(self): return [SimpleNamespace(name='images')]
            def run(self, _, inputs):
                self.tensor = inputs['images']
                # 640x320 input -> 384x192 with 96px top/bottom letterboxing.
                return [np.array([[0, 60, 126, 180, 156, 0, .9]], dtype=np.float32)]
        model = Model()
        result = detection._detect(model, np.zeros((320, 640, 3), dtype=np.uint8), .25)
        self.assertEqual(model.tensor.shape, (1, 3, 384, 384))
        self.assertEqual(result[0]['bbox_xyxy'], [100, 50, 300, 100])

    def test_invalid_plate_model_output_is_rejected(self):
        for rows in (np.zeros((1, 6)), np.array([[0, 1, 2, 3, 4, 0, float('nan')]]),
                     np.array([[0, 1, 2, 3, 4, 0, 1.2]])):
            model = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')], run=lambda *args: [rows])
            with self.subTest(shape=rows.shape), self.assertRaises(detection.PlateError):
                detection._detect(model, np.zeros((32,64,3), dtype=np.uint8), .25)

    @staticmethod
    def probabilities(text, confidence=1.):
        result = np.full((1, 10, 37), (1-confidence)/36, dtype=np.float64)
        for i, char in enumerate(text.ljust(10, '_')):
            result[0, i, ocr._CONTRACT['alphabet'].index(char)] = confidence
        return result

    def test_ocr_read_uncertain_and_unreadable_contract(self):
        self.assertEqual(ocr._decode(self.probabilities('34TEST01'), .8)['text'], '34TEST01')
        uncertain = ocr._decode(self.probabilities('34TEST01', .7), .8)
        self.assertEqual(uncertain['status'], 'uncertain')
        self.assertIsNone(uncertain['text'])
        self.assertEqual(uncertain['candidate_text'], '34TEST01')
        for text in ('', '34_TE01'):
            result = ocr._decode(self.probabilities(text), .8)
            self.assertEqual(result['status'], 'unreadable')
            self.assertIsNone(result['text'])

    def test_ocr_rejects_invalid_probability_tensor(self):
        for raw in (np.zeros((1,10,37)), np.ones((1,370)), np.full((1,10,37), float('nan')), np.zeros((1,5))):
            with self.subTest(shape=raw.shape), self.assertRaises(detection.PlateError):
                ocr._decode(raw, .8)

    def test_manifest_scope_dimensions_and_time(self):
        with tempfile.TemporaryDirectory() as td, patch.object(ocr, 'CROP_ROOT', Path(td)):
            folder = Path(td)/'plates-test'
            folder.mkdir()
            path = folder/'crops.json'
            crop = {'crop_path':str(folder/'one.png'), 'source_sec':2., 'frame_index':60,
                    'bbox_xyxy':[10,20,110,60], 'width':100, 'height':40, 'confidence':.9}
            data = {'video_path':'/not-needed.mp4', 'effective_range':{'start_sec':2.,'end_sec':3.},
                    'crop_count':1, 'ocr_performed':False, 'crops':[crop]}
            path.write_text(json.dumps(data))
            self.assertEqual(ocr._load_manifest(path, 10)[2], data)
            for updates in ({'source_sec':3.}, {'width':99}, {'crop_path':str(Path(td)/'other.png')}):
                path.write_text(json.dumps({**data,'crops':[{**crop, **updates}]}))
                with self.subTest(updates=updates), self.assertRaises(detection.PlateError):
                    ocr._load_manifest(path, 10)

    def test_zero_crops_does_not_load_ocr_model(self):
        with tempfile.TemporaryDirectory() as td, patch.object(ocr, 'CROP_ROOT', Path(td)/'plates'), \
                patch.object(ocr, 'OUTPUT_ROOT', Path(td)/'ocr'), \
                patch.object(ocr, '_settings', return_value=(.8, 500, 120)), patch.object(ocr, '_load_model') as loader:
            folder = Path(td)/'plates'/'plates-empty'
            folder.mkdir(parents=True)
            path = folder/'crops.json'
            path.write_text(json.dumps({'video_path':'/missing-source.mp4', 'effective_range':{'start_sec':0.,'end_sec':1.},
                                        'crop_count':0, 'ocr_performed':False, 'crops':[]}))
            result, warnings = ocr.read_plate_crops(str(path))
            loader.assert_not_called()
            self.assertFalse(result['ocr_performed'])
            self.assertEqual(result['processed_crop_count'], 0)
            self.assertIn('NO_CROPS', [w['code'] for w in warnings])
            self.assertTrue(Path(result['details_path']).is_file())
