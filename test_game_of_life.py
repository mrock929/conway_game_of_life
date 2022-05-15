# Tests for the game_of_life.py file

import numpy as np
import pytest
from pytest_mock import MockFixture
from typing import Union

from game_of_life import GameOfLife


def test_init(mocker: MockFixture):
    # Arrange
    board = np.zeros((5, 7))
    board[3, 4] = 1
    board_shape = np.shape(board)
    prefix = "test"

    plotting = mocker.patch.object(GameOfLife, "create_plot")
    checking = mocker.patch.object(GameOfLife, "check_board_values")

    # Act
    game = GameOfLife(
        board_array=board, show_plots=False, save_plots=True, file_prefix=prefix
    )

    # Assert
    assert game.height == board_shape[0]
    assert game.width == board_shape[1]
    assert checking.call_count == 1
    assert plotting.call_count == 1
    assert game.board[3, 4] == board[3, 4]
    assert game.prefix == prefix
    assert game.show_plots is False
    assert game.save_plots is True


@pytest.mark.parametrize("value, test_int", [(1.75, 1), (-5, 2), (6, 3)])
def test_check_board_values(
    value: Union[float, int], test_int: int, mocker: MockFixture
):
    # Arrange
    mocker.patch.object(GameOfLife, "__init__", return_value=None)
    game = GameOfLife()
    board = np.zeros((4, 5))
    board[1, 2] = value

    # Act / Assert
    if test_int == 1:
        with pytest.raises(TypeError, match="not all integers"):
            game.check_board_values(input_board=board)
    elif test_int == 2:
        with pytest.raises(ValueError, match="less than 0"):
            game.check_board_values(input_board=board)
    elif test_int == 3:
        with pytest.raises(ValueError, match="greater than 1"):
            game.check_board_values(input_board=board)


def test_propagate_board(mocker: MockFixture):
    # Arrange
    check_board = mocker.patch.object(GameOfLife, "check_board_values")
    find_ind = mocker.patch.object(GameOfLife, "find_chunk_indices")
    alive = mocker.patch.object(GameOfLife, "is_alive")
    plots = mocker.patch.object(GameOfLife, "create_plot")

    width = 4
    height = 2
    num_steps = 2
    board = np.zeros((height, width))
    game = GameOfLife(
        board_array=board, show_plots=True, save_plots=False, file_prefix="test"
    )

    # Act
    game.propagate_board(n_steps=num_steps)

    # Assert
    assert check_board.call_count == 1  # from init
    assert find_ind.call_count == num_steps * width * height
    assert alive.call_count == num_steps * width * height
    assert plots.call_count == num_steps + 1  # +1 from init


@pytest.mark.parametrize(
    "current_i, current_j, max_i, max_j, low_i, high_i, low_j, high_j, cell_i, cell_j",
    [
        (2, 1, 5, 3, 1, 4, 0, 3, 1, 1),
        (0, 0, 5, 3, 0, 2, 0, 2, 0, 0),
        (5, 3, 5, 3, 4, 6, 2, 4, 1, 1),
    ],
)
def test_find_chunk_indices(
    current_i: int,
    current_j: int,
    max_i: int,
    max_j: int,
    low_i: int,
    high_i: int,
    low_j: int,
    high_j: int,
    cell_i: int,
    cell_j: int,
    mocker: MockFixture,
):
    # Arrange
    mocker.patch.object(GameOfLife, "__init__", return_value=None)
    game = GameOfLife()

    # Act
    output = game.find_chunk_indices(
        current_i=current_i, current_j=current_j, max_i=max_i, max_j=max_j
    )

    # Assert
    assert isinstance(output, dict)
    assert output["low_i"] == low_i
    assert output["high_i"] == high_i
    assert output["low_j"] == low_j
    assert output["high_j"] == high_j
    assert output["cell_i"] == cell_i
    assert output["cell_j"] == cell_j


@pytest.mark.parametrize(
    "chunk, cell_i, cell_j, out_value",
    [
        (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]), 1, 1, 0),
        (np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]), 1, 1, 1),
        (np.array([[0, 1], [1, 1]]), 0, 0, 1),
        (np.array([[0, 0], [1, 1]]), 1, 1, 0),
    ],
)
def test_is_alive(
    chunk: np.array, out_value: int, cell_i: int, cell_j: int, mocker: MockFixture
):
    # Arrange
    mocker.patch.object(GameOfLife, "__init__", return_value=None)
    game = GameOfLife()

    # Act
    output = game.is_alive(chunk=chunk, cell_i=cell_i, cell_j=cell_j)

    # Assert
    assert output == out_value
