# Shim to preserve old entrypoint for run_bertscore_demo
# Original implementation moved to CXRMetric/metrics/bertscore/run_bertscore_demo.py
import os
import runpy
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
new_path = os.path.join(repo_root, 'CXRMetric', 'metrics', 'bertscore', 'run_bertscore_demo.py')
if not os.path.exists(new_path):
    print(f"Moved demo not found at {new_path}. Please run the new script directly from CXRMetric/metrics/bertscore.")
    sys.exit(1)

print("Note: run_bertscore_demo has moved to CXRMetric/metrics/bertscore/run_bertscore_demo.py — executing the relocated script now.")
runpy.run_path(new_path, run_name='__main__')
