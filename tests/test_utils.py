# -*- coding: utf-8 -*-
"""Tests for the denku.utils module."""

from denku.utils import (
    get_datetime,
    split_on_chunks,
    get_linear_value,
    get_cosine_value,
    get_ema_value
)


def test_get_datetime():
    """Test the get_datetime function."""
    dt = get_datetime()
    assert isinstance(dt, str)
    assert len(dt) > 0


def test_split_on_chunks():
    """Test the split_on_chunks function."""
    data = list(range(10))
    chunks = list(split_on_chunks(data, 3))
    assert len(chunks) == 3  # np.array_split returns 3 chunks for 10 elements

    # Convert numpy arrays to lists for comparison
    chunks = [chunk.tolist() for chunk in chunks]

    assert len(chunks[0]) == 4  # First chunk has 4 elements
    assert len(chunks[1]) == 3  # Second chunk has 3 elements
    assert len(chunks[2]) == 3  # Third chunk has 3 elements
    assert chunks[0] == [0, 1, 2, 3]
    assert chunks[-1] == [7, 8, 9]


def test_get_linear_value():
    """Test the get_linear_value function."""
    # Test with 3 arguments (current_index, start_value, total_steps)
    assert get_linear_value(5, 10, 10) == 5.0
    assert get_linear_value(0, 10, 10) == 10.0
    assert get_linear_value(10, 10, 10) == 0.0

    # Test with 4 arguments (current_index, start_value, total_steps, end_value)
    assert get_linear_value(5, 10, 10, 0) == 5.0
    assert get_linear_value(0, 10, 10, 0) == 10.0
    assert get_linear_value(10, 10, 10, 0) == 0.0


def test_get_cosine_value():
    """Test the get_cosine_value function."""
    # Test with 3 arguments (current_index, start_value, total_steps)
    assert get_cosine_value(0, 10, 10) == 10.0
    assert get_cosine_value(10, 10, 10) == 0.0
    assert get_cosine_value(5, 10, 10) == 5.0

    # Test with 4 arguments (current_index, start_value, total_steps, end_value)
    assert get_cosine_value(0, 10, 10, 0) == 10.0
    assert get_cosine_value(10, 10, 10, 0) == 0.0
    assert get_cosine_value(5, 10, 10, 0) == 5.0


def test_get_ema_value():
    """Test the get_ema_value function."""
    # Test with different values
    assert get_ema_value(0, 10, 0.5) == 10.0  # At start
    assert get_ema_value(1, 10, 0.5) == 5.0   # After one step
    assert get_ema_value(2, 10, 0.5) == 2.5   # After two steps
    assert get_ema_value(10, 10, 0.5) == 0.009765625  # After ten steps
