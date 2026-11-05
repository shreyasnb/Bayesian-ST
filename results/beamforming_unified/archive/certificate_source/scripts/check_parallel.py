"""Check scientific determinism and exact checkpoint restoration."""
import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.experiments import run_case1
from pandas.testing import assert_frame_equal
from pathlib import Path

def check():
    cfg=dict(seed=99,repeats=1,T=12,draws=60,burnin=40,max_draws=60,max_burnin=40)
    one=run_case1(**cfg,workers=1)
    temporary=tempfile.TemporaryDirectory(prefix='array-checkpoints-')
    directory=temporary.name
    two=run_case1(**cfg,workers=2,checkpoint_dir=directory)
    restored=run_case1(**cfg,workers=1,checkpoint_dir=directory)
    for key in ('metrics','regions','tuning'):
        scientific=[c for c in one[key].columns if not c.endswith('_seconds')]
        assert_frame_equal(one[key][scientific],two[key][scientific],check_exact=True)
        assert_frame_equal(two[key],restored[key],check_exact=True)
    files=list(Path(directory).glob('*.json'))
    assert len(files)==7
    temporary.cleanup()
    print('PASS: workers1/2 scientific scalars identical; JSON resume restores all values exactly; seven checkpoint files')

if __name__=='__main__':check()
