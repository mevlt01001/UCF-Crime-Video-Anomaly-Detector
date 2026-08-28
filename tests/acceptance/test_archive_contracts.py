from pathlib import Path
import json
import tempfile
from unittest.mock import patch
from support import OfflineCase
from utils import clip_archive as archive


class ArchiveContracts(OfflineCase):
    """MAN-A01/A02/A03/A04: real ledger/filesystem, fake FFmpeg."""
    def setUp(self):
        super().setUp()
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'ORIGINAL-DO-NOT-CHANGE')
        self.enterContext(patch.object(archive, 'ROOT', self.root / 'archive'))
        def export(source, target, start, end, **kwargs):
            Path(target).write_bytes(b'FAKE-CLIP-FOR-LEDGER-TEST-NOT-PLAYABLE')
        self.export = self.enterContext(patch.object(archive, 'export_video', side_effect=export))

    def call(self, category='belirsiz', explanation='Görüntüde olay türü belirsiz.'):
        return archive.archive_clip(str(self.source), 2., 5., category, explanation)

    def test_repeat_reuses_record_and_first_explanation(self):
        first = self.call()
        second = self.call(explanation='Farklı gerekçe')
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])
        self.assertEqual(first['output_path'], second['output_path'])
        self.assertEqual(first['explanation'], second['explanation'])
        self.assertEqual(self.export.call_count, 1)
        self.assertEqual(self.source.read_bytes(), b'ORIGINAL-DO-NOT-CHANGE')

    def test_same_clip_cannot_be_silently_recategorized(self):
        first = self.call()
        with self.assertRaises(archive.ArchiveError) as cm:
            self.call(category='trafik_kazasi')
        self.assertEqual(cm.exception.code, 'ARCHIVE_CATEGORY_CONFLICT')
        self.assertTrue(Path(first['output_path']).is_file())
        self.assertEqual(self.export.call_count, 1)

    def test_corrupt_archive_is_not_overwritten(self):
        first = self.call()
        path = Path(first['output_path'])
        path.write_bytes(b'BROKEN')
        with self.assertRaises(archive.ArchiveError) as cm:
            self.call()
        self.assertEqual(cm.exception.code, 'ARCHIVE_CONFLICT')
        self.assertEqual(path.read_bytes(), b'BROKEN')

    def test_bad_inputs_do_not_export(self):
        for category, reason in (('unknown', 'test'), ('belirsiz', ''), ('belirsiz', 'x' * 2001)):
            with self.subTest(category=category, reason_length=len(reason)), self.assertRaises(archive.ArchiveError):
                self.call(category, reason)
        self.export.assert_not_called()

    def test_failed_export_cleans_only_new_folder(self):
        first = self.call()
        def fail(*args, **kwargs):
            Path(args[1]).write_bytes(b'PARTIAL')
            raise RuntimeError('injected encoder failure')
        self.export.side_effect = fail
        with self.assertRaises(RuntimeError):
            archive.archive_clip(str(self.source), 6., 7., 'diger', 'Yeni olay')
        self.assertTrue(Path(first['output_path']).exists())
        self.assertEqual(list((self.root / 'archive' / 'diger').iterdir()), [])

    def test_incident_subfolder_groups_multiple_clips(self):
        first = archive.archive_clip(str(self.source), 2., 5., 'belirsiz', 'İlk kesit.', incident_id='olay_1')
        second = archive.archive_clip(str(self.source), 6., 7., 'belirsiz', 'İkinci kesit.', incident_id='olay_1')
        self.assertEqual(Path(first['incident_path']).parent.name, 'olay_1')
        self.assertEqual(Path(first['incident_path']).parents[2].name, 'belirsiz')
        self.assertEqual(first['incident_path'], second['incident_path'])
        manifest = json.loads(Path(first['incident_path']).read_text(encoding='utf-8'))
        self.assertEqual(manifest['incident_id'], 'olay_1')
        self.assertEqual(len(manifest['clips']), 2)

    def test_same_incident_name_in_different_videos_stays_separate(self):
        legacy = self.root / 'archive' / 'belirsiz' / 'olay_1' / 'incident.json'
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"legacy": true}')
        other = self.root / 'other.mp4'
        other.write_bytes(b'other source')
        first = archive.archive_clip(str(self.source), 2., 5., 'belirsiz', 'İlk video.', incident_id='olay_1')
        second = archive.archive_clip(str(other), 2., 5., 'belirsiz', 'İkinci video.', incident_id='olay_1')
        self.assertNotEqual(first['incident_path'], second['incident_path'])
        self.assertEqual(legacy.read_text(), '{"legacy": true}')
        for result, source in [(first, self.source), (second, other)]:
            manifest = json.loads(Path(result['incident_path']).read_text())
            self.assertEqual(manifest['video_path'], str(source))
            self.assertEqual(len(manifest['clips']), 1)
        again = archive.archive_clip(str(self.source), 2., 5., 'belirsiz', 'İlk video.', incident_id='olay_1')
        self.assertTrue(again['cache_hit'])
        with self.assertRaises(archive.ArchiveError):
            archive.archive_clip(str(self.source), 2., 5., 'soygun', 'Farklı kategori.', incident_id='olay_1')
