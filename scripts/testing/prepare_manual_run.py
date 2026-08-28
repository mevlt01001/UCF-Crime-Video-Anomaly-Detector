"""Create an evidence folder; inspect one local video without uploading/analyzing it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New folder under _stuff/test_runs; must not exist')
    args = parser.parse_args()
    with (ROOT/'docs/testing/sonuc_kaydi.csv').open() as f:
        known = {row['scenario_id'] for row in csv.DictReader(f)}
    if args.scenario not in known:
        parser.error('Unknown scenario ID')
    source = args.video.expanduser().resolve()
    if not source.is_file():
        parser.error('Video file not found')
    output = args.out.resolve()
    allowed = (ROOT/'_stuff/test_runs').resolve()
    if output == allowed or allowed not in output.parents:
        parser.error('--out must be a new subfolder of _stuff/test_runs')
    if output.exists():
        parser.error('Output already exists; choose another run suffix. Nothing overwritten.')
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    metadata = {}
    try:
        import cv2
        cap = cv2.VideoCapture(str(source))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            metadata = {'opened': bool(cap.isOpened()), 'fps': fps, 'frame_count': count,
                        'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        'duration_sec_estimate': count/fps if fps > 0 else None}
        finally:
            cap.release()
    except Exception as exc:
        metadata = {'metadata_error': str(exc)}
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True)
    record = {'scenario_id': args.scenario, 'status':'CALISTIRILMADI',
              'created_utc':datetime.now(timezone.utc).isoformat(), 'commit':commit.stdout.strip(),
              'python':sys.version.split()[0], 'platform':platform.platform(),
              'video_path':str(source), 'video_sha256':digest.hexdigest(), 'metadata':metadata,
              'ui_url':None, 'session_id':None, 'job_id':None,
              'observed':None, 'expected_vs_actual':None,
              'note':'Preparation only. No analysis/model/API request. No credentials collected.'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (output/'notes.md').write_text(f'# {args.scenario}\n\nDurum: CALISTIRILMADI\n\n'
                                 '## Ben ne yaptım?\n\n## Ekranda ne oldu?\n\n'
                                 '## Beklenen ile farkı\n\n## Paylaşmadan önce özel veri kontrolü\n')
    print(str(output))


if __name__ == '__main__':
    main()
