from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from common import run, COL_B, SEED

grid = {"model__hidden_layer_sizes": [(7,), (10,), (15,), (20,)],
        "model__alpha": [0.005, 0.01, 0.02, 0.05]}
model = MLPClassifier(solver="lbfgs", max_iter=100, random_state=SEED)
run("nnet", model, COL_B, grid, scaler=MinMaxScaler)