"""Value against price, and the value-vs-price quadrant."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl

from athletevalue.uncertainty.draws import FloatArray


class Quadrant(StrEnum):
    UNDERVALUED = "undervalued"
    """Worth more to the program than the market price."""
    STAR = "fair_star"
    """High value and high price."""
    OVERVALUED = "overvalued"
    """Priced above the value this model finds."""
    LOW_PRIORITY = "low_priority"


def surplus_draws(program_value: FloatArray, price: FloatArray) -> FloatArray:
    if program_value.shape != price.shape:
        raise ValueError("program value and price draws must have the same shape")
    result: FloatArray = program_value - price
    return result


def quadrant(value: float, price: float, *, value_cut: float, price_cut: float) -> Quadrant:
    high_value = value >= value_cut
    high_price = price >= price_cut
    if high_value and high_price:
        return Quadrant.STAR
    if high_value:
        return Quadrant.UNDERVALUED
    if high_price:
        return Quadrant.OVERVALUED
    return Quadrant.LOW_PRIORITY


def assign_quadrants(table: pl.DataFrame) -> pl.DataFrame:
    """Add a quadrant column using the median value and price of paid players as cuts."""
    paid = table.filter(pl.col("price") > 0)
    if paid.is_empty():
        return table.with_columns(pl.lit(Quadrant.LOW_PRIORITY.value).alias("quadrant"))
    value_cut = float(np.median(paid["program_value"].to_numpy()))
    price_cut = float(np.median(paid["price"].to_numpy()))
    labels = [
        quadrant(v, p, value_cut=value_cut, price_cut=price_cut).value
        for v, p in zip(table["program_value"], table["price"], strict=True)
    ]
    return table.with_columns(pl.Series("quadrant", labels, dtype=pl.Utf8))
