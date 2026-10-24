"""Calendar features retain valid mixed formats and reject ambiguous timezone mixes."""
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


def _processor():
    return build_fold_preprocessor(
        pd.DataFrame({'when': ['2024-01-01', '2024-01-02', '2024-01-03']}),
        PreprocessRecipe(impute=None, dates=True, date_columns=('when',), date_include_time=True),
    )


def test_valid_mixed_date_formats_keep_known_calendar_values():
    processor = _processor()
    held = pd.DataFrame({'when': ['2024-02-29 14:15:16', 'March 1, 2024 04:05:06']}, index=[19, 72])
    before = held.copy(deep=True)
    result = processor.transform(held)
    assert result['when_year'].tolist() == [2024, 2024]
    assert result['when_month'].tolist() == [2, 3]
    assert result['when_day'].tolist() == [29, 1]
    assert result['when_dayofyear'].tolist() == [60, 61]
    assert result['when_hour'].tolist() == [14, 4]
    assert result['when_minute'].tolist() == [15, 5]
    assert result['when_second'].tolist() == [16, 6]
    pd.testing.assert_frame_equal(held, before)


def test_calendar_values_do_not_depend_on_other_rows_in_the_batch():
    processor = _processor()
    frame = pd.DataFrame({'when': ['March 1, 2024', '2024-02-29']}, index=[3, 8])
    batch = processor.transform(frame)
    for index in frame.index:
        pd.testing.assert_frame_equal(processor.transform(frame.loc[[index]]), batch.loc[[index]])


def test_mixed_timezone_offsets_require_explicit_normalization():
    processor = _processor()
    frame = pd.DataFrame({'when': ['2024-01-01T10:00:00+00:00', '2024-06-01T10:00:00+01:00']})
    with pytest.raises(ValidationError, match='timezone'):
        processor.transform(frame)


def test_date_autodetection_accepts_valid_mixed_training_formats():
    train = pd.DataFrame({'when': ['2024-01-01', 'January 2, 2024', '2024-01-03']})
    processor = build_fold_preprocessor(train, PreprocessRecipe(impute=None, dates=True))
    assert processor.transform(train)['when_day'].tolist() == [1, 2, 3]
