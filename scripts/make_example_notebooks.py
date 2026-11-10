"""Compatibility entry point for editable notebook generation.

Plotting statements are emitted directly into each notebook cell. The old
pre-rendered-image notebook generator is retained in Git history and archive.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_editable_notebooks import generate

if __name__ == '__main__':
    generate(('01_array', '02_field'))
