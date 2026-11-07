"""Categorical effect sizes agree with independent statistical reference values."""
import itertools

import pandas as pd
import pytest
from scipy.stats.contingency import association

from buildml.eda.analyzers.bivariate import _cramers_v, analyze_bivariate


@pytest.mark.parametrize('counts', [
    [[20,0],[0,20]], [[10,10],[10,10]], [[4,1,5],[2,6,2]],
    [[8,1],[1,8],[4,2]],
])
def test_cramers_v_matches_independent_uncorrected_reference(counts):
    table = pd.DataFrame(counts)
    expected = association(table.to_numpy(), method='cramer', correction=False)
    assert _cramers_v(table) == pytest.approx(expected)
    assert _cramers_v(table.T) == pytest.approx(expected)
    rows = [(f'a{i}', f'b{j}') for i,j in itertools.product(range(len(counts)),
            range(len(counts[0]))) for _ in range(counts[i][j])]
    frame = pd.DataFrame(rows, columns=['a','b'])
    frame['ignored_id'] = [f'id{i}' for i in range(len(frame))]
    frame['target'] = frame.a
    result = analyze_bivariate(frame, target='target', feature_columns=['a','b','target'])
    assert result['feature_columns_analyzed'] == ['a','b']
    assert result['n_rows'] == len(frame)
    assert result['categorical_pairs'] == [{'a':'a', 'b':'b', 'cramers_v':pytest.approx(expected)}]
    assert not result['pearson']


@pytest.mark.parametrize('counts', [[], [[0,0],[0,0]], [[2,3]], [[2],[3]]])
def test_degenerate_tables_do_not_advertise_an_identifiable_effect(counts):
    assert _cramers_v(pd.DataFrame(counts)) is None


def test_categorical_screen_honors_documented_eight_column_cap():
    frame = pd.DataFrame({f'c{i}':['a','b']*10 for i in range(9)})
    result = analyze_bivariate(frame)
    pairs = result['categorical_pairs']
    assert len(pairs) == 28
    assert {name for pair in pairs for name in (pair['a'],pair['b'])} == set(frame.columns[:8])
    assert all(pair['cramers_v'] == pytest.approx(1.) for pair in pairs)
