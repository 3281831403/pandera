# pylint: disable=W0212
"""Tests for value-level semantic inference of string-loaded tables.

Covers the workflow of validating scraped, all-string batches with an
inferred schema: numeric/boolean/datetime columns are recognized by value,
cleaned through an attached parser, keep min/max range checks, stay
idempotent on already-normalized data, and reject genuine dirty values with
errors naming the column and row. Leading-zero identifiers and mixed text
remain strings, and strongly-typed tables keep their previous inference.
"""

import numpy as np
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.engines import pandas_engine
from pandera.errors import SchemaError, SchemaErrors
from pandera.schema_inference._semantic import (
    NULL_TOKENS,
    infer_semantic_type,
)
from pandera.schema_inference.pandas import (
    infer_dataframe_schema,
    infer_series_schema,
)


def _assert_dtype(column, alias: str) -> None:
    """Assert the column's dtype matches the given engine dtype alias."""
    expected = pandas_engine.Engine.dtype(alias)
    assert expected.check(column.dtype)


def _scraped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["0123", "0456", "0789"],
            "name": ["Widget A", "Widget B", "Gadget C"],
            "price": ["$1,200.50", "$99.99", "N/A"],
            "qty": ["10", "-", "3"],
            "in_stock": ["yes", "no", "YES"],
            "listed": [
                "2026-01-05",
                "2026-02-10",
                "2026-03-01",
            ],
        }
    )


def test_infers_semantic_types_for_string_columns() -> None:
    """Text columns are typed by value instead of falling back to object."""
    schema = infer_dataframe_schema(_scraped_frame())

    _assert_dtype(schema.columns["price"], "float64")
    _assert_dtype(schema.columns["qty"], "Int64")
    _assert_dtype(schema.columns["in_stock"], "boolean")
    _assert_dtype(schema.columns["listed"], "datetime64[ns]")

    # leading-zero identifiers and free text stay strings.
    _assert_dtype(schema.columns["sku"], "object")
    _assert_dtype(schema.columns["name"], "object")
    assert not schema.columns["sku"].parsers
    assert not schema.columns["name"].parsers

    # every converted column carries its cleaning parser.
    for col in ("price", "qty", "in_stock", "listed"):
        assert len(schema.columns[col].parsers) == 1


def test_validate_raw_batch_cleans_values() -> None:
    """The inferred schema validates the raw text batch directly."""
    raw = _scraped_frame()
    schema = infer_dataframe_schema(raw)
    cleaned = schema.validate(raw)

    assert cleaned["price"].tolist()[:2] == [1200.50, 99.99]
    assert pd.isna(cleaned.loc[2, "price"])
    assert cleaned["qty"].tolist() == [10, pd.NA, 3]
    assert cleaned["in_stock"].tolist() == [True, False, True]
    assert cleaned["listed"].tolist() == list(
        pd.to_datetime(raw["listed"]).tolist()
    )

    # placeholders imply nullability; columns without them stay non-nullable.
    assert schema.columns["price"].nullable
    assert schema.columns["qty"].nullable
    assert not schema.columns["in_stock"].nullable
    assert not schema.columns["listed"].nullable


def test_validation_of_normalized_data_is_idempotent() -> None:
    """Re-validating cleaned output must not change it."""
    raw = _scraped_frame()
    schema = infer_dataframe_schema(raw)
    cleaned = schema.validate(raw)
    again = schema.validate(cleaned)
    pd.testing.assert_frame_equal(cleaned, again)


@pytest.mark.parametrize(
    "column, dirty_value",
    [
        ("price", "sold out"),
        ("qty", "many"),
        ("in_stock", "maybe"),
        ("listed", "not-a-date"),
    ],
)
def test_dirty_values_fail_with_column_and_row(
    column: str, dirty_value: str
) -> None:
    """Unparseable values raise instead of silently becoming null."""
    raw = _scraped_frame()
    schema = infer_dataframe_schema(raw)
    raw.loc[1, column] = dirty_value

    with pytest.raises(SchemaError) as exc_info:
        schema.validate(raw, lazy=False)
    error = exc_info.value
    assert error.column_name == column
    assert list(error.failure_cases.index) == [1]
    assert error.failure_cases.iloc[0] == dirty_value

    with pytest.raises(SchemaErrors) as exc_info_lazy:
        schema.validate(raw, lazy=True)
    lazy_errors = exc_info_lazy.value.schema_errors
    assert len(lazy_errors) == 1
    assert lazy_errors[0].column_name == column
    assert lazy_errors[0].failure_cases.iloc[0] == dirty_value


def test_lazy_validation_collects_multiple_dirty_columns() -> None:
    """Lazy mode reports every unparseable column in one SchemaErrors."""
    raw = _scraped_frame()
    schema = infer_dataframe_schema(raw)
    raw.loc[0, "price"] = "sold out"
    raw.loc[2, "in_stock"] = "maybe"

    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(raw, lazy=True)
    columns = {err.column_name for err in exc_info.value.schema_errors}
    assert columns == {"price", "in_stock"}


def test_numeric_and_datetime_columns_keep_range_checks() -> None:
    """Min/max checks are inferred from the cleaned values."""
    raw = _scraped_frame()
    schema = infer_dataframe_schema(raw)

    registry_names = {
        check.registry_name for check in schema.columns["price"].checks
    }
    assert registry_names == {
        "greater_than_or_equal_to",
        "less_than_or_equal_to",
    }
    listed_checks = {
        check.registry_name for check in schema.columns["listed"].checks
    }
    assert listed_checks == {
        "greater_than_or_equal_to",
        "less_than_or_equal_to",
    }

    # the range check still rejects out-of-range, correctly-typed values.
    cleaned = schema.validate(raw)
    cleaned.loc[0, "price"] = 99999.0
    with pytest.raises(SchemaError):
        schema.validate(cleaned)


def test_strongly_typed_table_inference_is_unchanged() -> None:
    """Tables with real dtypes infer exactly as before (no parsers)."""
    typed = pd.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [1.0, 2.0, 3.0],
            "b": [True, False, True],
            "s": ["a", "b", "c"],
            "d": pd.to_datetime(["20180101", "20180102", "20180103"]),
        }
    )
    schema = infer_dataframe_schema(typed)
    _assert_dtype(schema.columns["i"], "int64")
    _assert_dtype(schema.columns["f"], "float64")
    _assert_dtype(schema.columns["b"], "bool")
    _assert_dtype(schema.columns["s"], "object")
    _assert_dtype(schema.columns["d"], "datetime64[ns]")
    for column in schema.columns.values():
        assert not column.parsers

    # validation round-trips and stays idempotent.
    validated = schema.validate(typed)
    pd.testing.assert_frame_equal(validated, schema.validate(validated))


def test_placeholder_variants_become_null() -> None:
    """All configured null spellings are recognized as missing values."""
    # "100" is neither a boolean token nor a null token.
    values = list(sorted(NULL_TOKENS - {""})) + ["100"]
    frame = pd.DataFrame({"qty": values})
    schema = infer_dataframe_schema(frame)
    _assert_dtype(schema.columns["qty"], "Int64")
    cleaned = schema.validate(frame)
    assert cleaned.loc[cleaned.index[-1], "qty"] == 100
    assert cleaned["qty"].iloc[:-1].isna().all()


def test_currency_and_grouping_formats_parse_as_numbers() -> None:
    frame = pd.DataFrame({"price": ["$1,234.56", "€99", "£1,000", "12.5"]})
    schema = infer_dataframe_schema(frame)
    cleaned = schema.validate(frame)
    assert cleaned["price"].tolist() == [1234.56, 99.0, 1000.0, 12.5]


def test_boolean_spellings_are_case_insensitive() -> None:
    frame = pd.DataFrame({"flag": ["yes", "NO", "True", "false", "Y", "n"]})
    schema = infer_dataframe_schema(frame)
    _assert_dtype(schema.columns["flag"], "boolean")
    cleaned = schema.validate(frame)
    assert cleaned["flag"].tolist() == [True, False, True, False, True, False]


def test_iso_datetimes_including_timezones() -> None:
    frame = pd.DataFrame(
        {"ts": ["2026-01-05T08:30:00Z", "2026-02-10T09:00:00+02:00"]}
    )
    schema = infer_dataframe_schema(frame)
    cleaned = schema.validate(frame)
    assert str(cleaned["ts"].dtype) == "datetime64[ns, UTC]"
    # idempotent on the tz-aware normalized output.
    pd.testing.assert_series_equal(
        cleaned["ts"], schema.validate(cleaned)["ts"]
    )


def test_non_iso_dates_remain_strings() -> None:
    frame = pd.DataFrame({"day": ["12-31-2026", "01/02/2026"]})
    schema = infer_dataframe_schema(frame)
    _assert_dtype(schema.columns["day"], "object")
    assert not schema.columns["day"].parsers


def test_leading_zero_and_mixed_text_remain_strings() -> None:
    frame = pd.DataFrame(
        {
            "code": ["0123", "0456", "0789"],
            "mixed": ["10", "abc", "3"],
            "percent": ["100%", "50%", "75%"],
        }
    )
    schema = infer_dataframe_schema(frame)
    for name in ("code", "mixed", "percent"):
        _assert_dtype(schema.columns[name], "object")
        assert not schema.columns[name].parsers
    cleaned = schema.validate(frame)
    pd.testing.assert_frame_equal(cleaned, frame)


def test_string_dtype_column_is_semantically_inferred() -> None:
    """pandas ``StringDtype`` columns get the same treatment as object ones."""
    frame = pd.DataFrame({"v": pd.array(["1", "2", None], dtype="string")})
    schema = infer_dataframe_schema(frame)
    _assert_dtype(schema.columns["v"], "Int64")
    assert schema.columns["v"].nullable
    cleaned = schema.validate(frame)
    assert cleaned["v"].tolist() == [1, 2, pd.NA]
    pd.testing.assert_frame_equal(cleaned, schema.validate(cleaned))


def test_series_schema_semantic_inference() -> None:
    series = pd.Series(["$1.50", "$2.25", "-"], name="price")
    schema = infer_series_schema(series)
    _assert_dtype(schema, "float64")
    assert len(schema.parsers) == 1
    cleaned = schema.validate(series)
    assert cleaned.iloc[0] == 1.50
    assert pd.isna(cleaned.iloc[2])
    pd.testing.assert_series_equal(cleaned, schema.validate(cleaned))


def test_placeholder_in_non_nullable_inference_fails() -> None:
    """A batch inferred without nulls rejects a later placeholder value."""
    schema = infer_dataframe_schema(pd.DataFrame({"n": ["1", "2"]}))
    assert not schema.columns["n"].nullable
    with pytest.raises(SchemaErrors):
        schema.validate(pd.DataFrame({"n": ["1", "N/A"]}), lazy=True)


def test_semantic_inference_on_empty_and_typed_series() -> None:
    # all-placeholder columns have no inferable semantics and stay strings.
    placeholders = pd.Series(["N/A", "-", None], name="x")
    result = infer_semantic_type(placeholders)
    assert result.kind == "string"
    assert result.parser is None

    # a typed series is left untouched.
    typed = pd.Series([1, 2, 3], name="x")
    assert infer_semantic_type(typed).parser is None


def test_numeric_parser_idempotent_on_nullable_types() -> None:
    schema = infer_dataframe_schema(pd.DataFrame({"n": ["1", "N/A", "3"]}))
    cleaned = schema.validate(pd.DataFrame({"n": ["1", "N/A", "3"]}))
    # Int64 with pd.NA must survive a second validation unchanged.
    again = schema.validate(cleaned)
    pd.testing.assert_frame_equal(cleaned, again)
    assert again["n"].dtype == pd.Int64Dtype()


def test_index_is_preserved() -> None:
    frame = pd.DataFrame(
        {"v": ["1", "2"]}, index=pd.Index([10, 11], name="idx")
    )
    schema = infer_dataframe_schema(frame)
    assert schema.index.name == "idx"
    cleaned = schema.validate(frame)
    assert cleaned.index.name == "idx"
    assert cleaned.index.tolist() == [10, 11]


def test_min_max_bounds_use_cleaned_values() -> None:
    frame = pd.DataFrame({"v": ["$10.00", "$20.00", "$30.00"]})
    schema = infer_dataframe_schema(frame)
    bounds = {
        check.registry_name: check.statistics
        for check in schema.columns["v"].checks
    }
    assert bounds["greater_than_or_equal_to"]["min_value"] == 10.0
    assert bounds["less_than_or_equal_to"]["max_value"] == 30.0


def test_integer_like_values_infer_integral_dtype() -> None:
    integral = infer_dataframe_schema(pd.DataFrame({"v": ["10", "20", "30"]}))
    _assert_dtype(integral.columns["v"], "Int64")

    fractional = infer_dataframe_schema(pd.DataFrame({"v": ["10.5", "20"]}))
    _assert_dtype(fractional.columns["v"], "float64")


def test_numpy_nan_and_none_inputs_are_handled() -> None:
    frame = pd.DataFrame(
        {"v": pd.Series(["1", np.nan, None, "3"], dtype=object)}
    )
    schema = infer_dataframe_schema(frame)
    cleaned = schema.validate(frame)
    assert cleaned["v"].tolist() == [1, pd.NA, pd.NA, 3]
    pd.testing.assert_frame_equal(cleaned, schema.validate(cleaned))
