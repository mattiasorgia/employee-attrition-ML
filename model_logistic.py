from sklearn.linear_model import LogisticRegression
from common import run, COL_A

run("logistic", LogisticRegression(max_iter=1000), COL_A)