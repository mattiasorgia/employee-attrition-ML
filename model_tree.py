import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from common import run, COL_A, SEED, OUT

grid = {"model__ccp_alpha": np.arange(0.00001, 0.05, 0.002)}   # come cp_grid in R
search = run("tree", DecisionTreeClassifier(random_state=SEED), COL_A, grid)

best = search.best_estimator_
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best["model"], feature_names=best["prep"].get_feature_names_out(),
          class_names=["Stayed", "Left"], max_depth=3, filled=True, ax=ax)
fig.savefig(OUT / "tree.png", dpi=150, bbox_inches="tight")
plt.close(fig)