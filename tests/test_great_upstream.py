from pathlib import Path
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks.common.great import GREAT


FIXTURES = Path(__file__).parent / 'fixtures'
CASES = json.loads((FIXTURES / 'gerost_great_cases.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_matches_unchanged_reference_in_octave(case):
    with np.load(FIXTURES / 'gerost_great.npz', allow_pickle=False) as fixture:
        name = case['name']
        initial = fixture[name + '_initial']
        observations = fixture[name + '_observations']
        kwargs = {} if case['default_controls'] else dict(
            stepsize=case['stepsize'], inner_steps=case['inner_steps'])
        tracker = GREAT(case['n'], case['k'], window=case['window'],
                        initial_basis=initial, complex_data=case['complex_data'], **kwargs)
        steps = 0
        for index, observation in enumerate(observations.T):
            tracker.update(observation)
            assert_allclose(tracker.projector, fixture[name + '_projectors'][index],
                            rtol=3e-11, atol=3e-12)
            assert_allclose(tracker.covariance, fixture[name + '_covariances'][index],
                            rtol=3e-13, atol=3e-13)
            expected_step = fixture[name + '_stepsizes'][index, 0]
            if np.isnan(expected_step):
                assert tracker.last_stepsize is None
                assert_allclose(tracker.basis, initial, rtol=0, atol=0)
            else:
                assert_allclose(tracker.last_stepsize, expected_step, rtol=3e-13, atol=0)
                steps += case['inner_steps']
            assert tracker.total_gradient_steps == steps
