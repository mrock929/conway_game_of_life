# Code for generating and visualizing Conway's Game of Life: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from typing import Optional


class GameOfLife:
    """Class to implement and visualize Conway's Game of Life"""

    def __init__(
        self,
        board_array: np.array,
        show_plots: Optional[bool] = False,
        save_plots: Optional[bool] = True,
        file_prefix: Optional[str] = "plot",
    ):
        """
        Initialize class and board

        Args:
            board_array: Starting state of the board
            show_plots: If true, display plots to the user
            save_plots: If true, save plots to file
            file_prefix: Prefix for saving plots to file, default is "plot"
        """

        # Check to ensure board is valid
        self.check_board_values(input_board=board_array)

        board_shape = np.shape(board_array)

        self.height = board_shape[0]
        self.width = board_shape[1]
        self.board = board_array
        self.prefix = file_prefix
        self.show_plots = show_plots
        self.save_plots = save_plots

        if show_plots or save_plots:
            self.create_plot(step=0)

    @staticmethod
    def check_board_values(input_board: np.array) -> None:
        """
        Check to make sure the values of the board are all 1s and 0s

        Args:
            input_board: Input board state

        Returns: None

        Raises:
            TypeError if something other than ints are in the board state
            ValueError if something other than 1s and 0s are in the board state

        """

        if not np.all(np.equal(np.mod(input_board, 1), 0)):
            raise TypeError(f"Initial board elements are not all integers.")

        if input_board.max() > 1:
            raise ValueError(f"Initial board contains an integer greater than 1")

        if input_board.min() < 0:
            raise ValueError(f"Initial board contains an integer less than 0")

    def propagate_board(self, n_steps: int) -> None:
        """
        Propagate the board forward n_steps. Triggers the display or saving of plots based on inputs.

        Args:
            n_steps: Number of steps to propagate the board

        Returns: Nothing, but shows plots if show_plots == True

        """

        for board_step in np.arange(
            1, n_steps + 1, 1
        ):  # start steps at 1 so 0 is initial state

            # Initialize board for next step
            new_board = np.zeros_like(self.board)

            for i in range(self.height):
                for j in range(self.width):
                    # Note: behavior will be very different for periodic vs non-periodic boundary conditions.
                    # Assume non-periodic for now.
                    # Extract indices for the relevant part of the board
                    indices = self.find_chunk_indices(
                        current_i=i,
                        current_j=j,
                        max_i=self.height - 1,
                        max_j=self.width - 1,
                    )
                    # Determine if the cell of interest is alive in the next step
                    new_board[i, j] = self.is_alive(
                        chunk=deepcopy(
                            self.board[
                                indices["low_i"] : indices["high_i"],
                                indices["low_j"] : indices["high_j"],
                            ]
                        ),
                        cell_i=indices["cell_i"],
                        cell_j=indices["cell_j"],
                    )

            # Update board state
            self.board = new_board

            if self.show_plots or self.save_plots:
                self.create_plot(step=board_step)

    def create_plot(self, step: int) -> None:
        """
        Create plots for display and saving

        Args:
            step: Which iteration step the current image is for

        Returns: Nothing, but shows or saves plots if needed

        """

        # Create plot
        plt.figure()
        plt.imshow(self.board, cmap="gray_r")
        plt.title(f"Step {step}")
        plt.tick_params(axis="both", labelsize=0, length=0)

        if self.show_plots and self.save_plots:
            plt.savefig(Path(f"./plots/{self.prefix}_{step}"))
            plt.show()
            plt.close()
        elif self.show_plots:
            plt.show()
            plt.close()
        elif self.save_plots:
            plt.savefig(Path(f"./plots/{self.prefix}_{step}"))
            plt.close()

    @staticmethod
    def find_chunk_indices(
        current_i: int, current_j: int, max_i: int, max_j: int
    ) -> dict:
        """
        Find the indices for extracting the current chunk of the board for determining if the cell of interest is alive.
        Need to grab all adjacent (orthogonal or diagonal) cells

        Args:
            current_i: Current cell of interest y (rows) index
            current_j: Current cell of interest x (columns) index
            max_i: Maximum value for y (rows) index
            max_j: Maximum value for x (columns) index

        Returns: Dict with the min_i, max_i range of row indices, the min_j, max_j range of column indices, and a dict
            with the local integers within the returned chunk for the cell of interest

        """

        low_i = max(current_i - 1, 0)
        high_i = min(current_i + 1, max_i)
        low_j = max(current_j - 1, 0)
        high_j = min(current_j + 1, max_j)

        # Add one to high values to grab the relevant parts of the array
        high_i += 1
        high_j += 1

        # Determine the chunk relative indices of cell of interest
        if low_i == 0 and high_i - low_i == 2:
            cell_i = 0  # top side of chunk with height 2
        else:
            cell_i = 1
        if low_j == 0 and high_j - low_j == 2:
            cell_j = 0  # top of chunk with height 2
        else:
            cell_j = 1  # middle of chunk with height 3

        return {
            "low_i": low_i,
            "high_i": high_i,
            "low_j": low_j,
            "high_j": high_j,
            "cell_i": cell_i,
            "cell_j": cell_j,
        }

    @staticmethod
    def is_alive(chunk: np.array, cell_i: int, cell_j: int) -> int:
        """
        Checks if the cell at cell_index of the chunk should be alive (1) or dead (0)

        Args:
            chunk: The numpy array containing the cell of interest and all relevant surrounding cells
            cell_i: The row index within the chunk of the cell of interest
            cell_j: The column index within the chunk of the cell of interest

        Returns: Whether the cell of interest is alive (1) or dead (0)

        """

        # Determine cell start state since this changes the resulting rules
        if chunk[cell_i, cell_j] == 1:
            alive = True
            # Set cell of interest to 0 the rules don't take into account the current cell
            chunk[cell_i, cell_j] = 0
        else:
            alive = False

        # Total number of relevant nearby alive cells
        total_alive = np.sum(chunk)

        # Determine if cell of interest is alive or dead. See wiki page above for rules
        if alive:
            if total_alive < 2 or total_alive > 3:
                return 0
            else:
                return 1
        else:
            if total_alive == 3:
                return 1
            else:
                return 0
