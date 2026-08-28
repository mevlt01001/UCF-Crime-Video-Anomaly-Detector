"""Run an explicit test group and retain its real exit code plus full local logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['existing','contracts','gaps','frontend'])
    args = parser.parse_args()
    groups = {
        'existing': [[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v']],
        'contracts': [[sys.executable,'-B','-m','unittest','discover','-s','tests/acceptance','-p','test_*.py','-v']],
        'gaps': [[sys.executable,'-B','-m','unittest','discover','-s','tests/acceptance','-p','gap_*.py','-v']],
        'frontend': [['npm','--prefix','frontend','run','lint'],['npm','--prefix','frontend','run','build']],
    }
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output = ROOT/'_stuff/test_runs'/f'{args.suite}-{tag}'
    output.mkdir(parents=True, exist_ok=False)
    cache = output/'cache'
    cache.mkdir()
    env = dict(os.environ, MPLCONFIGDIR=str(cache), XDG_CACHE_HOME=str(cache))
    records = []
    print(f'Result folder: {output}', flush=True)
    if args.suite == 'gaps':
        print('Known-gap suite: assertion failures are expected on the current implementation.', flush=True)
    code = 0
    with (output/'output.log').open('w',encoding='utf-8') as log:
        for command in groups[args.suite]:
            line = '$ ' + ' '.join(command) + '\n'
            print(line,end='',flush=True)
            log.write(line)
            try:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = 130
            except OSError as exc:
                log.write(str(exc)+'\n')
                print(str(exc),file=sys.stderr)
                code = 127
            records.append({'command':command,'exit_code':code})
            if code:
                break
    (output/'summary.json').write_text(json.dumps({'suite':args.suite,'commands':records,
        'exit_code':code, 'note':'Check output.log; an import error is not reproduction of a known bug.'},indent=2)+'\n')
    print(f'Exit code: {code}\nShare: {output / "output.log"}\nSummary: {output / "summary.json"}')
    return code if code >= 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
