from sklearn.ensemble import RandomForestClassifier
from common import run, COL_A, SEED

grid = {"model__max_features": ["sqrt", 0.3, 0.5]}   # 3 valori, come tuneLength = 3
run("rf", RandomForestClassifier(n_estimators=500, random_state=SEED), COL_A, grid)