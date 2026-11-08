"""Execute in place; write outputs even on failure and propagate the exception."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
p=Path(sys.argv[1]).resolve()
nb=nbformat.read(p,as_version=4)
try:
    NotebookClient(nb,timeout=int(os.environ.get('BAYESIAN_ST_NOTEBOOK_TIMEOUT','3600')),kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    nbformat.write(nb,p)
