"""Repeat real buffered DDP validation; retain every failure without retries."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distributed-debug', choices=('OFF', 'INFO', 'DETAIL'), default='INFO')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    report = {
        'status': 'running', 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'scope': 'two CPU Gloo ranks, BatchNorm buffers, train/validation forwards; no deployment SLA',
        'requested_repetitions': args.repetitions, 'attempts': [],
        'distributed_debug': args.distributed_debug,
    }
    environment = {**os.environ, 'PYTHONFAULTHANDLER': '1',
                   'TORCH_SHOW_CPP_STACKTRACES': '1', 'TORCH_DISTRIBUTED_DEBUG': args.distributed_debug}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for iteration in range(1, args.repetitions + 1):
        started = time.monotonic()
        xml = args.output.resolve() / f'attempt-{iteration}.xml'
        with (args.output / f'attempt-{iteration}.log').open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest',
                    'tests/unit/test_ddp_result_transfer.py::test_validation_with_buffers_completes_on_all_ranks',
                    '-q', '--no-cov', '--tb=long', '--junitxml=' + str(xml),
                ], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=660)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        suites = list(ET.parse(xml).getroot().iter('testsuite')) if xml.exists() else []
        executed = sum(int(s.attrib.get('tests', 0)) - int(s.attrib.get('skipped', 0)) for s in suites)
        passed = code == 0 and executed == 1
        report['attempts'].append({'iteration': iteration, 'exit_code': code,
                                   'executed': executed, 'seconds': time.monotonic() - started,
                                   'passed': passed})
        report['status'] = 'running' if passed else 'failed'
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report['attempts'][-1]), flush=True)
        if not passed:
            return 1
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
