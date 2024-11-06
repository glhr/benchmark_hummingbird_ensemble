import numpy as np
import time
from hummingbird.ml import convert

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class EnsembleMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble_models=[]):
        
        self.ensemble_models = ensemble_models
        self.ensemble_n = len(ensemble_models)
        
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y
        for n in range(self.ensemble_n):
            self.ensemble_models[n].fit(X,y)
        self.is_fitted_ = True
        return self
    
    def _preds_from_ensemble(self, X):
        preds = []
        for n in range(self.ensemble_n):
            preds.append(self.ensemble_models[n].predict_proba(X))
        return np.stack(preds)

def convert_sklearn_ensemble_to_hummingbird(ensemble_model, mode="CPU"):
    ensemble_members = ensemble_model.ensemble_models
    ensemble_n = ensemble_model.ensemble_n

    ensemble_members_hb = []
    for n,ensemble_member in enumerate(ensemble_members):
        ensemble_members_hb.append((str(n),ensemble_member))

    mlp = StackingClassifier(
            estimators=ensemble_members_hb, final_estimator=None, passthrough=False,
            stack_method="predict_proba", cv="prefit"
        )

    mlp.final_estimator_ = "identity"
    mlp.estimators_ = ensemble_members
    mlp.stack_method_ = ["predict_proba" for _ in range(ensemble_n)]
    predict_func = lambda m,x: m.model.forward(x).reshape(x.shape[0],-1,2).mean(axis=1).cpu().numpy()

    model = convert(mlp, 'pytorch')
    if mode == "GPU": model.to('cuda')

    return model, predict_func

@ignore_warnings(category=ConvergenceWarning) # for this benchmark we don't care if the model converges
def time_sklearn_model(input_size=(100000,5),hidden_layer_sizes=(64,64),activation="logistic",
                       alpha=0, ensemble_n=100,mode="CPU",**kwargs):

    # some fake training data. the training does not matter for this benchmark
    X_train = np.zeros((4,input_size[1])).astype(np.float32)
    Y_train = np.array([0,1,0,1]).astype(np.int32)
    fit_iter = 1

    ensemble_models = []
    for n in range(ensemble_n):
        ensemble_models.append(
            MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,max_iter=fit_iter,verbose=False).fit(X_train, Y_train)
        )
    mlp = EnsembleMLPClassifier(ensemble_models=ensemble_models)

    model, predict_func = convert_sklearn_ensemble_to_hummingbird(mlp, mode=mode)
    
    n_predictions = 0
    start_time = time.perf_counter()
    while n_predictions < 1000:
        y = predict_func(model,X_train)[:,1]
        duration_secs = (time.perf_counter() - start_time)
        n_predictions += 1
    throughput = (n_predictions*input_size[0]) / duration_secs
    return throughput

if __name__ == "__main__":
    mode = "CPU"
    print(f"Running benchmark with {mode}")

    print(f"...Warmup...")
    throughput = time_sklearn_model(ensemble_n=100,input_size=(100000,5), mode=mode) # warmup
    throughput = time_sklearn_model(ensemble_n=100,input_size=(100000,3), mode=mode) # warmup
    
    throughput = time_sklearn_model(ensemble_n=100,input_size=(100000,5), mode=mode) # parallel converter case (5 dimensions)
    print(f"Parallel converter case - Throughput: {throughput} predictions/sec")

    throughput = time_sklearn_model(ensemble_n=100,input_size=(100000,3), mode=mode) # single converter case (3 dimensions)
    print(f"Single converter case - Throughput: {throughput} predictions/sec")

    