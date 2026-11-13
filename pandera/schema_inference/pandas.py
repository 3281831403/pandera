"""Module for inferring dataframe/series schema."""

from typing import overload

import pandas as pd

from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.components import Column, Index, MultiIndex
from pandera.api.pandas.container import DataFrameSchema
from pandera.schema_inference._semantic import (
    SemanticInference,
    infer_semantic_type,
)
from pandera.schema_statistics.pandas import (
    infer_dataframe_statistics,
    infer_series_statistics,
    parse_check_statistics,
)


@overload
def infer_schema(
    pandas_obj: pd.Series,
) -> SeriesSchema:  # pragma: no cover
    ...


@overload
def infer_schema(  # type: ignore[misc]
    pandas_obj: pd.DataFrame,
) -> DataFrameSchema:  # pragma: no cover
    ...


def infer_schema(pandas_obj):
    """Infer schema for pandas DataFrame or Series object.

    :param pandas_obj: DataFrame or Series object to infer.
    :returns: DataFrameSchema or SeriesSchema
    :raises: TypeError if pandas_obj is not expected type.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        return infer_dataframe_schema(pandas_obj)
    elif isinstance(pandas_obj, pd.Series):
        return infer_series_schema(pandas_obj)
    else:
        raise TypeError(
            "pandas_obj type not recognized. Expected a pandas DataFrame or "
            f"Series, found {type(pandas_obj)}"
        )


def _create_index(index_statistics):
    index = [
        Index(
            properties["dtype"],
            checks=parse_check_statistics(properties["checks"]),
            nullable=properties["nullable"],
            name=properties["name"],
        )
        for properties in index_statistics
    ]
    if len(index) == 1:
        index = index[0]  # type: ignore
    else:
        index = MultiIndex(index)  # type: ignore

    return index


def _semantic_check_statistics(
    inference: SemanticInference,
) -> dict | None:
    """Min/max checks for a semantically inferred column, matching the
    statistics produced for strongly-typed numeric/datetime columns."""
    cleaned = inference.cleaned
    if cleaned.isna().all():
        return None
    if inference.kind == "datetime":
        return {
            "greater_than_or_equal_to": cleaned.min(),
            "less_than_or_equal_to": cleaned.max(),
        }
    if inference.kind == "numeric":
        return {
            "greater_than_or_equal_to": float(cleaned.min()),
            "less_than_or_equal_to": float(cleaned.max()),
        }
    return None


def infer_dataframe_schema(
    df: pd.DataFrame, *, infer_str_length: bool = False
) -> DataFrameSchema:
    """Infer a DataFrameSchema from a pandas DataFrame.

    Beyond dtype-level inference, all-string (object/``string`` dtype) columns
    are inspected by value: columns whose non-null values consistently parse
    as numbers, booleans (``yes``/``no``, ``true``/``false``, ...) or ISO
    datetimes are inferred with that dtype together with a parser that
    performs the equivalent cleaning (``N/A``/``-``/... placeholders become
    null, currency and thousands separators are stripped, etc.). The inferred
    schema therefore validates the raw text batch directly, while validating
    an already-normalized batch is idempotent. Values that cannot be parsed
    make validation fail with a
    :class:`~pandera.errors.ParserError` naming the column and rows instead
    of being silently coerced to null. Columns that do not parse cleanly
    (mixed text, leading-zero identifiers such as ``"0123"``) remain string
    columns.

    :param df: DataFrame object to infer.
    :param infer_str_length: also infer ``str_length`` checks for string-like
        columns. Off by default so that inferred schemas keep validating when
        a column's dtype is later updated (e.g. together with a parser).
    :returns: DataFrameSchema
    """
    df_statistics = infer_dataframe_statistics(
        df, infer_str_length=infer_str_length
    )
    columns = {}
    for colname, properties in df_statistics["columns"].items():
        inference = infer_semantic_type(df[colname])
        if inference.parser is None:
            columns[colname] = Column(
                properties["dtype"],
                checks=parse_check_statistics(properties["checks"]),
                nullable=properties["nullable"],
            )
            continue

        nullable = bool(df[colname].isna().any()) or bool(
            inference.cleaned.isna().any()
        )
        columns[colname] = Column(
            inference.dtype,
            checks=parse_check_statistics(
                _semantic_check_statistics(inference)
            ),
            parsers=[inference.parser],
            nullable=nullable,
        )

    schema = DataFrameSchema(
        columns=columns,
        index=_create_index(df_statistics["index"]),
        coerce=True,
    )
    return schema


def infer_series_schema(
    series, *, infer_str_length: bool = False
) -> SeriesSchema:
    """Infer a SeriesSchema from a pandas DataFrame.

    Applies the same value-level inference as
    :func:`infer_dataframe_schema`: text series that consistently parse as
    numbers, booleans or ISO datetimes get the corresponding dtype and a
    cleaning parser; anything else stays a plain string series.

    :param series: Series object to infer.
    :param infer_str_length: also infer ``str_length`` checks for string-like
        values (see :func:`infer_dataframe_schema`).
    :returns: SeriesSchema
    """
    series_statistics = infer_series_statistics(
        series, infer_str_length=infer_str_length
    )
    inference = infer_semantic_type(series)
    if inference.parser is None:
        return SeriesSchema(
            dtype=series_statistics["dtype"],
            checks=parse_check_statistics(series_statistics["checks"]),
            nullable=series_statistics["nullable"],
            name=series_statistics["name"],
            coerce=True,
        )

    nullable = bool(series.isna().any()) or bool(
        inference.cleaned.isna().any()
    )
    return SeriesSchema(
        dtype=inference.dtype,
        checks=parse_check_statistics(_semantic_check_statistics(inference)),
        parsers=[inference.parser],
        nullable=nullable,
        name=series_statistics["name"],
        coerce=True,
    )
