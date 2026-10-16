from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from common import run, make_preprocessor, X_fit, y_fit, COL_B, cv, METRIC, SEED

Xs = X_fit[COL_B].sample(10000, random_state=SEED)
ys = y_fit.loc[Xs.index]

pipe = Pipeline([("prep", make_preprocessor(Xs)), ("model", KNeighborsClassifier())])
tune = GridSearchCV(pipe, {"model__n_neighbors": list(range(25, 70, 2))},
                    cv=cv, scoring=METRIC, n_jobs=-1)
tune.fit(Xs, ys)
best_k = tune.best_params_["model__n_neighbors"]
print("k scelto:", best_k)

run("knn", KNeighborsClassifier(n_neighbors=best_k), COL_B)