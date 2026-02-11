import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Big, sparse-signal problem
# -----------------------------
# Key: HUGE dimensionality, tiny number of informative features,
# no label noise, reasonable class separation.
X, y = make_classification(
    n_samples=100000,       # big-ish, still manageable
    n_features=200,       # many features -> overfitting risk
    n_informative=10,      # tiny true signal
    n_redundant=10,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    class_sep=1.2,         # fairly separable if you keep the right features
    flip_y=0.08,            # no label noise, so best C can get very high acc
    weights=[0.5, 0.5],
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features for better convergence and to make C more meaningful
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Helper function to train and evaluate a single trial
def train_and_evaluate(trial_params):
    """Train and evaluate logistic regression with a specific C value."""
    i, C = trial_params
    
    clf = LogisticRegression(
        C=C,
        penalty="l1",
        solver="saga",      # supports L1 on large, high-dim problems
        max_iter=3000,
        n_jobs=1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    return i, C, acc

# Main execution with MPI
if __name__ == '__main__':
    # Generate all trial parameters
    rng = np.random.default_rng(0)
    n_trials = 32
    
    trial_params = []
    for i in range(n_trials):
        # log-uniform C in [1e-5, 1e+2]
        C = 10 ** rng.uniform(-5, -1)
        trial_params.append((i, C))
    
    start_time = time.time()
    
    # Run trials in parallel using MPIPoolExecutor
    with MPIPoolExecutor() as executor:
        futures = [executor.submit(train_and_evaluate, params) for params in trial_params]
        
        results = []
        for future in futures:
            i, C, acc = future.result()
            print(f"{i:02d}: C={C:.3e}, acc={acc:.4f}")
            results.append((C, acc))
    
    finish_time = time.time()
    
    # Find best C
    best_C, best_acc = max(results, key=lambda t: t[1])
    print("\nBest result:")
    print(f"C={best_C:.3e}, acc={best_acc:.4f}")
    print(f"Total time execution: {finish_time-start_time:.4f}s")