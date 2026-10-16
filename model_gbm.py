from sklearn.ensemble import GradientBoostingClassifier
from common import run, COL_A, SEED

grid = {"model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 5],
        "model__learning_rate": [0.01, 0.1]}
run("gbm", GradientBoostingClassifier(min_samples_leaf=10, random_state=SEED), COL_A, grid)