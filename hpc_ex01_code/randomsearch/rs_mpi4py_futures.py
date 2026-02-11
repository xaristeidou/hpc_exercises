import numpy as np
import time
from mpi4py import MPI
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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate all trial parameters (only on rank 0, then broadcast)
rng = np.random.default_rng(0)
n_trials = 32

if rank == 0:
    trial_params = []
    for i in range(n_trials):
        # log-uniform C in [1e-5, 1e+2]
        C = 10 ** rng.uniform(-5, -1)
        trial_params.append((i, C))
    start_time = time.time()
else:
    trial_params = None

# Broadcast all trial parameters to all ranks
trial_params = comm.bcast(trial_params, root=0)

# Distribute trials: each rank processes a subset
local_trials = [trial_params[i] for i in range(len(trial_params)) if i % size == rank]

# Process local trials
local_results = []
for i, C in local_trials:
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
    
    local_results.append((i, C, acc))

# Gather all results to rank 0
all_results = comm.gather(local_results, root=0)

# Process results on rank 0
if rank == 0:
    results = []
    for rank_results in all_results:
        results.extend(rank_results)
    
    # Sort by trial index for consistent output
    results.sort(key=lambda x: x[0])
    
    print("# Single-parameter HPO on C (log-uniform random search)")
    print("# Parallelized with MPI")
    for i, C, acc in results:
        print(f"{i:02d}: C={C:.3e}, acc={acc:.4f}")
    
    # Find best C
    best_C, best_acc = max([(C, acc) for _, C, acc in results], key=lambda t: t[1])
    finish_time = time.time()
    
    print("\nBest result:")
    print(f"C={best_C:.3e}, acc={best_acc:.4f}")
    print(f"Total time execution: {finish_time-start_time:.4f}s")
    print(f"MPI processes used: {size}")