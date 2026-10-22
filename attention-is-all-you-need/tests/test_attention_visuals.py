"""Known answers for the numerical examples used in the matrix displays."""

import math

import pytest
import torch

pytest.importorskip("matplotlib")
pytest.importorskip("ipywidgets")

from attention_visuals import (
    matrix_product_data,
    output_projection_example,
    projection_example,
)


def test_rectangular_matrix_multiplication():
    data = matrix_product_data(
        [[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]
    )
    expected = torch.tensor([[58., 64.], [139., 154.]], dtype=torch.float64)
    torch.testing.assert_close(data["result"], expected)


def test_projection_matrices_and_scores():
    data = projection_example()
    expected = {
        "Q": [[1, 0], [1, 1], [2, 1]],
        "K": [[1, 1], [0, -1], [1, 0]],
        "V": [[2.5, -.5], [.5, 2.5], [2.5, 2.5]],
        "K_T": [[1, 0, 1], [1, -1, 0]],
        "raw_scores": [[1, 0, 1], [2, -1, 1], [3, -1, 2]],
    }
    for name, values in expected.items():
        torch.testing.assert_close(data[name], torch.tensor(values, dtype=torch.float64))


def test_softmax_row_and_value_mix_have_hand_calculated_answers():
    data = projection_example()
    alpha = math.exp(1 / math.sqrt(2))
    denominator = 2 * alpha + 1
    expected_weights = torch.tensor(
        [alpha, 1, alpha], dtype=torch.float64
    ) / denominator
    expected_context = torch.tensor(
        [(5 * alpha + .5) / denominator, (2 * alpha + 2.5) / denominator],
        dtype=torch.float64,
    )
    torch.testing.assert_close(data["A"][0], expected_weights)
    torch.testing.assert_close(data["context"][0], expected_context)


def test_head_concatenation_and_output_projection():
    data = output_projection_example()
    torch.testing.assert_close(
        data["merged"],
        torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        data["output"],
        torch.tensor([[3, 1, 2, 2], [1, 3, 2, 2]], dtype=torch.float64),
    )


def test_bias_broadcasts_across_rows():
    data = matrix_product_data([[1, 2], [3, 4]], [[1, 0], [0, 1]], [.5, -.5])
    torch.testing.assert_close(
        data["result"],
        torch.tensor([[1.5, 1.5], [3.5, 3.5]], dtype=torch.float64),
    )


@pytest.mark.parametrize("example", [projection_example, output_projection_example])
def test_display_examples_preserve_model_initialization_random_state(example):
    before = torch.random.get_rng_state().clone()
    example()
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "left,right,bias",
    [([[1, 2]], [[1, 2]], None), ([1, 2], [[1], [2]], None),
     ([[1, 2]], [[1], [2]], [0, 1])],
)
def test_invalid_matrix_shapes_raise(left, right, bias):
    with pytest.raises(ValueError):
        matrix_product_data(left, right, bias)
