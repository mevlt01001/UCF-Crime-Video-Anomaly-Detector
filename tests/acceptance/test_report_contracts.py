import json
from support import OfflineCase, VIDEO, OTHER_VIDEO, evidence, pair, report
from utils.action_records import action_records
from utils.reporting import validate_report


class ReportContracts(OfflineCase):
    """MAN-R01/R02/R03/R04, MAN-E01/E02/E03/E04: recorded evidence only."""
    def test_coverage_can_be_union_of_adjacent_visual_calls(self):
        messages = evidence(visuals=((2., 3.), (3., 5.)))
        actions = action_records(messages, VIDEO)
        self.assertEqual(validate_report(json.dumps(report(actions=actions)), messages, VIDEO), report(actions=actions))

    def test_gap_in_visual_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report()), evidence(visuals=((2., 3.), (3.2, 5.))), VIDEO)

    def test_wrong_video_cannot_fill_coverage(self):
        messages = evidence(visuals=()) + evidence(video=OTHER_VIDEO)
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report()), messages, VIDEO)

    def test_failed_vlm_cannot_fill_coverage(self):
        messages = evidence(visuals=()) + pair('analyze_video_with_vlm', {'video_path': VIDEO}, {
            'video_path': VIDEO, 'vlm_response': 'a result', 'effective_range': {'start_sec': 2., 'end_sec': 5.}}, ok=False)
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report()), messages, VIDEO)

    def test_event_outside_visual_range_or_duration_is_rejected(self):
        for second in (-1., 1., 12., float('nan'), float('inf')):
            with self.subTest(second=second), self.assertRaises(ValueError):
                validate_report(json.dumps(report(events=[{'olay_id': 'olay_1', 'saniye': second, 'aciklama': 'Gözlem'}])), evidence(), VIDEO)

    def test_missing_archive_for_report_event_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report()), evidence(archives=()), VIDEO)

    def test_cannot_invent_or_drop_action_record(self):
        messages = evidence()
        records = action_records(messages, VIDEO)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(validate_report(json.dumps(report(actions=records)), messages, VIDEO)['eylemler'], records)
        for entries in ([], [records[0] + ' değiştirildi'], ['[BASARILI] hayali kayıt']):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                validate_report(json.dumps(report(actions=entries)), messages, VIDEO)

    def test_optional_action_failure_does_not_block_complete_visual_report(self):
        messages = evidence(archives=()) + pair('archive_anomaly_clip', {
            'video_path': VIDEO, 'start_sec': 2., 'end_sec': 5.,
            'category': 'belirsiz', 'explanation': 'Görsel kanıt yetersiz.', 'incident_id': 'olay_1'}, ok=False, code='FFMPEG_NOT_FOUND')
        records = action_records(messages, VIDEO)
        self.assertIn('[BASARISIZ]', records[0])
        validate_report(json.dumps(report(actions=records)), messages, VIDEO)

    def test_actions_preserve_result_order_and_duplicate_results_are_ignored(self):
        a = pair('save_video_segment', {'video_path': VIDEO}, {'video_path': VIDEO}, call_id='a')
        b = pair('archive_anomaly_clip', {'video_path': VIDEO, 'start_sec': 2., 'end_sec': 5.,
                                          'category': 'belirsiz', 'explanation': 'x', 'incident_id': 'olay_1'},
                 {'video_path': VIDEO}, call_id='b')
        records = action_records(a + b + [a[-1]], VIDEO)
        self.assertEqual(len(records), 2)
        self.assertIn('(a)', records[0])
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report(actions=records[::-1])), evidence() + a + b, VIDEO)

    def test_orphan_pending_and_wrong_video_actions_are_excluded(self):
        a = pair('archive_anomaly_clip', {'video_path': VIDEO, 'start_sec': 2., 'end_sec': 5.,
                                          'category': 'belirsiz', 'explanation': 'x', 'incident_id': 'olay_1'},
                 {'video_path': VIDEO})
        b = pair('save_video_segment', {'video_path': OTHER_VIDEO}, {'video_path': OTHER_VIDEO})
        for messages in ([a[0]], [a[1]], b):
            self.assertEqual(action_records(messages, VIDEO), [])

    def test_ocr_requires_manifest_from_same_task_detection(self):
        detection = pair('detect_license_plate_regions', {'video_path': VIDEO}, {
            'video_path': VIDEO, 'details_path': '/acceptance/plates/crops.json', 'crop_count': 0})
        reading = pair('read_license_plate_crops', {'crops_manifest_path': '/acceptance/plates/crops.json'}, {
            'video_path': VIDEO, 'ocr_performed': False})
        self.assertEqual(action_records(reading, VIDEO), [])
        records = action_records(detection + reading, VIDEO)
        self.assertEqual(len(records), 2)
        self.assertIn('OCR çalıştırılmadı', records[-1])
        self.assertNotIn('OCR tamamlandı', records[-1])

    def test_suggestions_do_not_count_as_executed_actions(self):
        messages = evidence()
        records = action_records(messages, VIDEO)
        for text in ('[ONERI] Arşivlenen klip insan denetimine gönderilmelidir.',
                     '[ONERI] Başarısız ikinci kesit tekrar değerlendirilsin.'):
            self.assertEqual(
                validate_report(json.dumps(report(actions=records + [text])), messages, VIDEO)['eylemler'],
                records + [text],
            )
        for text in ('[ONERI] ', '[ONERI] ' + 'x' * 2001, 'Kaydedildi'):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                validate_report(json.dumps(report(actions=records + [text])), messages, VIDEO)

    def test_mismatched_incident_id_blocks_report(self):
        messages = evidence(archives=[('olay_2', 2., 5., 'belirsiz')])
        with self.assertRaises(ValueError):
            validate_report(json.dumps(report()), messages, VIDEO)
