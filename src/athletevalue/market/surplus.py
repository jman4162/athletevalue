"""Value against price, and where a player sits among his teammates."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl

from athletevalue.uncertainty.draws import FloatArray


class Quadrant(StrEnum):
    """Position against the medians of paid teammates. A description, not a verdict."""

    VALUE_ABOVE_PRICE = "value_above_price"
    """Program value above the team median; price below it."""
    BOTH_ABOVE_MEDIAN = "both_above_median"
    VALUE_BELOW_PRICE = "value_below_price"
    """Price above the team median; program value below it."""
    BOTH_BELOW_MEDIAN = "both_below_median"


def surplus_draws(program_value: FloatArray, price: FloatArray) -> FloatArray:
    if program_value.shape != price.shape:
        raise ValueError("program value and price draws must have the same shape")
    result: FloatArray = program_value - price
    return result


def quadrant(value: float, price: float, *, value_cut: float, price_cut: float) -> Quadrant:
    high_value = value >= value_cut
    high_price = price >= price_cut
    if high_value and high_price:
        return Quadrant.BOTH_ABOVE_MEDIAN
    if high_value:
        return Quadrant.VALUE_ABOVE_PRICE
    if high_price:
        return Quadrant.VALUE_BELOW_PRICE
    return Quadrant.BOTH_BELOW_MEDIAN


def assign_quadrants(table: pl.DataFrame) -> pl.DataFrame:
    """Add quadrant, value_cut and price_cut columns.

    Cuts are the medians of program value and price among players the allocation
    pays. Unpaid players get no quadrant: the comparison is meaningless for them.
    """
    paid = table.filter(pl.col("price") > 0)
    if paid.is_empty():
        return table.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("quadrant"),
            pl.lit(None, dtype=pl.Float64).alias("value_cut"),
            pl.lit(None, dtype=pl.Float64).alias("price_cut"),
        )
    value_cut = float(np.median(paid["program_value"].to_numpy()))
    price_cut = float(np.median(paid["price"].to_numpy()))
    labels = [
        quadrant(v, p, value_cut=value_cut, price_cut=price_cut).value if p > 0 else None
        for v, p in zip(table["program_value"], table["price"], strict=True)
    ]
    return table.with_columns(
        pl.Series("quadrant", labels, dtype=pl.Utf8),
        pl.lit(value_cut).alias("value_cut"),
        pl.lit(price_cut).alias("price_cut"),
    )
