"""Project coverage must record the work performed by real Python worker threads."""
import json
import subprocess
import sys
from pathlib import Path


def test_project_tracer_records_worker_function_body(tmp_path):
    project = Path(__file__).resolve().parents[2]
    probe = tmp_path / 'probe_module.py'
    probe.write_text('def calculate(value):\n    doubled = 2 * value\n    return doubled + 3\n')
    driver = tmp_path / 'driver.py'
    driver.write_text('''import concurrent.futures
import coverage
import sys
c=coverage.Coverage(config_file=sys.argv[1], source=['probe_module'], data_file='.probe-coverage')
c.start()
import probe_module
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    assert list(pool.map(probe_module.calculate, [1,2,3])) == [5,7,9]
c.stop()
c.save()
c.json_report(outfile='observed.json')
''')
    subprocess.run([sys.executable, str(driver), str(project / 'pyproject.toml')], cwd=tmp_path,
                   check=True, capture_output=True, text=True, timeout=30)
    observed = json.loads((tmp_path / 'observed.json').read_text())
    recorded = next(v for k, v in observed['files'].items() if k.endswith('probe_module.py'))
    assert {2, 3}.issubset(recorded['executed_lines']), recorded
