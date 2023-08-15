from pathlib import Path

import numpy as np

DATA_NAME = "dataset900_a(-30.0,-30.0,-30.0,)_(30.0,30.0,30.0,)_b(44.0,66.0,84.0,)_(66.0,99.0,126.0,)"

X_TRAIN_PATH = Path().cwd() / DATA_NAME / "train" / "X_data.csv"
X_TEST_PATH = Path().cwd() / DATA_NAME / "test" / "X_data.csv"

X_TRAIN_MM = np.loadtxt(X_TRAIN_PATH, delimiter=",")[:, :3]
X_TEST_MM = np.loadtxt(X_TEST_PATH, delimiter=",")[:, :3]
