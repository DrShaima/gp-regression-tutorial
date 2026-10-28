"""Run this project's trusted plain-Python notebook and save its outputs.

This lightweight runner does not start a Jupyter kernel or support magics.
"""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import base64
import io
import json
import os
import platform

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
path = ROOT / 'gp_regression_tutorial.ipynb'
notebook = json.loads(path.read_text())
namespace = {'__name__': '__main__'}
count = 0
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    outputs = []
    stream = io.StringIO()
    def capture_show(*args, **kwargs):
        value = stream.getvalue()
        if value:
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': value})
            stream.seek(0)
            stream.truncate(0)
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format='png', dpi=130, bbox_inches='tight')
            outputs.append({'output_type': 'display_data', 'metadata': {},
                            'data': {'image/png': base64.b64encode(buffer.getvalue()).decode(),
                                     'text/plain': 'Matplotlib figure'}})
        plt.close('all')
    plt.show = capture_show
    with redirect_stdout(stream), redirect_stderr(stream):
        exec(compile(''.join(cell['source']), f'notebook-cell-{count}', 'exec'), namespace)
    if stream.getvalue():
        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
    cell['execution_count'] = count
    cell['outputs'] = outputs
notebook['metadata']['language_info']['version'] = platform.python_version()
path.write_text(json.dumps(notebook, indent=1) + '\n')
print(f'Executed {count} code cells successfully: {path.name}')
for row in namespace['rows']:
    print(row)
