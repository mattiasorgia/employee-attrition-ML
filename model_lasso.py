import numpy as np
from sklearn.linear_model import LogisticRegression
from common import run, COL_A, X_fit

lambdas = np.arange(0.001, 0.1, 0.005)          # come seq(0.001, 0.1, by = 0.005) in R
grid = {"model__C": 1 / (lambdas * len(X_fit))}  # C = 1 / (lambda * n)
run("lasso", LogisticRegression(l1_ratio=1, solver="liblinear"), COL_A, grid)