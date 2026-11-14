import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_contact_transport_diagnostics(root):
    root = Path(root)
    config = json.loads((root / 'configs/contact.json').read_text())['config']
    table = pd.read_csv(root / 'results/contact/data/checkpoint_summary.csv')
    curves = []
    for method in ('mean', 'mode', 'iso', 'mean_identity'):
        rows = table[table.method.eq(method)].sort_values('t')
        np.testing.assert_array_equal(rows.t, np.arange(4, 193, 4))
        curves.append(dict(method=method, points=rows[['t','mean','lower','upper']].to_numpy()))
    return dict(config=config, sphere_curves=curves)
