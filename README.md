# conway_game_of_life
Python implementation of Conway's Game of Life. See https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life for a summary of the game.

See `run_game_of_life.ipynb` for examples running `game_of_life.py`.

# Input Parameters

- `board_array`: A numpy array of the initial board state. Can be any size.
- `show_plots`: Optional, default is False. If True, shows the plots in the Jupyter notebook. It is not recommended to use True outside of Jupyter.
- `save_plots`: Optional, default is True. If True, saves the plots into /plots.
- `file_prefix`: Optional, default is `"plot"`. A string that specifies the file prefix to use when saving files. An underscore and the step number will be appended to the prefix for each saved figure.  
