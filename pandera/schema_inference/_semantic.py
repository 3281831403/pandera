"""Value-level semantic type inference for string-loaded pandas data.

HTML tables and similar text sources arrive as all-string ``object`` columns:
prices carry ``$`` and thousands separators, missing values are written as
``N/A`` or ``-``, booleans are spelled ``yes``/``no`` and dates follow ISO
8601. :func:`pandas.api.types.infer_dtype` only sees "string" for such columns,
so the regular dtype inference leaves them as ``object`` with no checks.

This module inspects the *values* of an object/string series and decides
whether the whole column parses as numeric, boolean or datetime, producing:

* the target :class:`~pandera.dtypes.DataType`,
* a :class:`~pandera.api.parsers.Parser` that applies the equivalent cleaning
  (placeholders -> null, formatted text -> typed values) and raises a
  :class:`~pandera.errors.ParserError` naming the unparseable rows instead of
  silently coercing them to null,
* the cleaned series, used to infer the same min/max checks the ordinary
  inference attaches to strongly-typed numeric/datetime columns.

Columns that do not parse cleanly (mixed text, leading-zero identifiers) stay
strings, preserving the previous inference result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pandera import dtypes
from pandera.api.parsers import Parser
from pandera.engines import numpy_engine, pandas_engine
from pandera.errors import ParserError

#: Spellings treated as missing values. Matched case-insensitively after
#: stripping surrounding whitespace.
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "n/a", "na", "null", "none", "nan", "-", "--", "#n/a", "?"}
)

#: Currency symbols stripped before numeric parsing.
_CURRENCY_SYMBOLS = "$€£¥₹"
_CURRENCY_PATTERN = re.compile(
    "|".join(re.escape(symbol) for symbol in _CURRENCY_SYMBOLS)
)

#: Accepted boolean spellings, mapped to their value.
_BOOL_TOKENS: dict[str, bool] = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "1": True,
    "0": False,
}

# Currency symbols and grouping characters stripped before numeric parsing.
_NUMERIC_PATTERN = re.compile(
    r"""
    ^\s*                # optional leading whitespace
    [+-]?               # optional sign
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?  # grouped digits: 1,234 or 1,234.5
        | \d+(?:\.\d+)?               # plain number: 12, 12.5
        | \.\d+                       # .5
    )
    (?:[eE][+-]?\d+)?  # optional exponent
    \s*$
    """,
    re.VERBOSE,
)
# Same grammar without the grouping commas, applied after they are stripped.
_NUMERIC_PATTERN_NO_GROUP = re.compile(
    r"^\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?\s*$"
)
# Non-capturing marker for a trailing timezone (Z or +/-HH:MM).
_TZ_MARKER = r"(?:Z|[+-]\d{2}:?\d{2})$"

# Strict ISO 8601 date or datetime, e.g. ``2026-01-05`` or
# ``2026-01-05T08:30:00Z``. Timezone offsets/``Z`` are accepted and converted
# to UTC.
_ISO_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


@dataclass(frozen=True)
class SemanticInference:
    """Result of value-level inference for one string series.

    :ivar kind: ``"numeric"``, ``"boolean"``, ``"datetime"`` or ``"string"``.
    :ivar dtype: target pandera dtype (``None`` when the column stays string).
    :ivar parser: cleaning parser, or ``None`` for plain string columns.
    :ivar cleaned: series with placeholders replaced by null and the remaining
        values converted to the inferred type (strings untouched otherwise).
    """

    kind: str
    dtype: Any
    parser: Parser | None
    cleaned: pd.Series


def _normalized_tokens(series: pd.Series) -> pd.Series:
    """String representation of non-null values, whitespace-stripped."""
    return series.dropna().astype(str).str.strip()


def _is_null_token(token: str) -> bool:
    return token.lower() in NULL_TOKENS


def _is_numeric_token(token: str) -> bool:
    if _is_null_token(token):
        return False
    # Leading-zero identifiers ("0123", "007") are codes, not numbers.
    if re.match(r"^[+-]?0\d+$", token):
        return False
    bare = _CURRENCY_PATTERN.sub("", token)
    return _NUMERIC_PATTERN.match(bare) is not None


def _is_boolean_token(token: str) -> bool:
    return token.lower() in _BOOL_TOKENS


def _is_datetime_token(token: str) -> bool:
    return _ISO_PATTERN.match(token) is not None


def _parser_error(
    series: pd.Series, invalid: pd.Series, semantic_type: str
) -> ParserError:
    rows = invalid.index.tolist()
    failure_cases = invalid.copy()
    return ParserError(
        f"cannot parse {len(invalid)} value(s) in column "
        f"{series.name!r} as {semantic_type}; "
        f"failing row label(s): {rows}, value(s): {invalid.tolist()}",
        failure_cases=failure_cases,
    )


def _make_numeric_parser() -> Parser:
    def parse_numeric(series: pd.Series) -> pd.Series:
        if not _is_text_series(series):
            return series

        tokens = series.astype("string")
        stripped = tokens.str.strip()
        is_null = stripped.str.lower().isin(NULL_TOKENS)
        is_na = series.isna() | is_null

        cleaned = (
            stripped.str.replace(",", "", regex=False)
            .str.replace(_CURRENCY_PATTERN, "", regex=True)
            .str.strip()
        )
        # Leading-zero identifiers must never be silently turned into numbers.
        leading_zero = stripped.str.match(r"^[+-]?0\d+$").fillna(False)
        numeric_mask = (
            cleaned.str.match(_NUMERIC_PATTERN_NO_GROUP).fillna(False)
            & ~leading_zero
        )

        invalid = series[~numeric_mask & ~is_na]
        if not invalid.empty:
            raise _parser_error(series, invalid, "numeric")

        result = pd.to_numeric(cleaned, errors="coerce")
        result[is_na] = pd.NA
        return result

    return Parser(parse_numeric, name="infer_numeric")


def _make_boolean_parser() -> Parser:
    def parse_boolean(series: pd.Series) -> pd.Series:
        if not _is_text_series(series):
            return series

        tokens = series.astype("string").str.strip()
        is_null = tokens.str.lower().isin(NULL_TOKENS)
        is_na = series.isna() | is_null

        lower = tokens.str.lower()
        bool_mask = lower.isin(_BOOL_TOKENS)
        invalid = series[~bool_mask & ~is_na]
        if not invalid.empty:
            raise _parser_error(series, invalid, "boolean")

        result = lower.map(_BOOL_TOKENS).astype("boolean")
        result[is_na] = pd.NA
        return result

    return Parser(parse_boolean, name="infer_boolean")


def _make_datetime_parser() -> Parser:
    def parse_datetime(series: pd.Series) -> pd.Series:
        if not _is_text_series(series):
            return series

        tokens = series.astype("string").str.strip()
        is_null = tokens.str.lower().isin(NULL_TOKENS)
        is_na = series.isna() | is_null

        iso_mask = tokens.str.match(_ISO_PATTERN).fillna(False)
        invalid = series[~iso_mask & ~is_na]
        if not invalid.empty:
            raise _parser_error(series, invalid, "datetime")

        has_tz = tokens.str.contains(_TZ_MARKER, regex=True).fillna(False)
        if has_tz.any():
            parsed = pd.to_datetime(
                tokens, errors="coerce", format="mixed", utc=True
            )
        else:
            parsed = pd.to_datetime(tokens, errors="coerce", format="mixed")
        parsed[is_na] = pd.NaT
        return parsed

    return Parser(parse_datetime, name="infer_datetime")


def _is_text_series(series: pd.Series) -> bool:
    """Whether the parser needs to run at all (text input).

    Text means an ``object`` dtype (the dtype of scraped string tables) or
    the pandas ``StringDtype``. Already-typed columns (numbers, booleans,
    datetimes) pass through unchanged, which is what makes re-validating
    normalized data idempotent.
    """
    dtype = pandas_engine.Engine.dtype(series.dtype)
    return dtypes.is_string(dtype) or isinstance(dtype, numpy_engine.Object)


def infer_semantic_type(series: pd.Series) -> SemanticInference:
    """Infer the value-level semantics of a pandas series.

    Non-text dtypes are returned unchanged (``kind="string"`` with no parser;
    callers keep their existing dtype-based inference). For text columns every
    non-null value must parse under one semantic type, otherwise the column
    stays a string.

    :param series: pandas Series to inspect.
    :returns: :class:`SemanticInference` describing the inferred type, its
        cleaning parser and the cleaned values.
    """
    if not _is_text_series(series) or series.dropna().empty:
        return SemanticInference(
            kind="string", dtype=None, parser=None, cleaned=series
        )

    tokens = _normalized_tokens(series)
    non_null = tokens[~tokens.map(_is_null_token)]
    if non_null.empty:
        # Only placeholders: nothing semantic can be inferred, keep string.
        return SemanticInference(
            kind="string", dtype=None, parser=None, cleaned=series
        )

    if non_null.map(_is_boolean_token).all():
        parser = _make_boolean_parser()
        cleaned = parser._parser_fn(series)
        return SemanticInference(
            kind="boolean",
            dtype=pandas_engine.Engine.dtype("boolean"),
            parser=parser,
            cleaned=cleaned,
        )

    if non_null.map(_is_numeric_token).all():
        parser = _make_numeric_parser()
        cleaned = parser._parser_fn(series)
        all_integral = False
        if not cleaned.dropna().empty:
            numeric = cleaned.dropna()
            all_integral = bool(
                pd.api.types.is_integer_dtype(numeric)
                or ((numeric % 1) == 0).all()
            )
        target_dtype = (
            pandas_engine.Engine.dtype("Int64")
            if all_integral
            else pandas_engine.Engine.dtype("float64")
        )
        return SemanticInference(
            kind="numeric",
            dtype=target_dtype,
            parser=parser,
            cleaned=cleaned,
        )

    if non_null.map(_is_datetime_token).all():
        parser = _make_datetime_parser()
        cleaned = parser._parser_fn(series)
        has_tz = non_null.str.contains(_TZ_MARKER, regex=True).any()
        target_dtype = (
            pandas_engine.Engine.dtype("datetime64[ns, UTC]")
            if has_tz
            else pandas_engine.Engine.dtype("datetime64[ns]")
        )
        return SemanticInference(
            kind="datetime",
            dtype=target_dtype,
            parser=parser,
            cleaned=cleaned,
        )

    return SemanticInference(
        kind="string", dtype=None, parser=None, cleaned=series
    )
