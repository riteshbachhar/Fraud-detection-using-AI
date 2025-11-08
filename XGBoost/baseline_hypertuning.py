# Import libraries
import pandas as pd
import polars as pl
from datetime import timedelta
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
import itertools
import pandas as pd

opt_features = ["Amount", "year", "day_of_month", "day_of_week", "hour", "minute", "second",
            "currency_mismatch", "high_risk_sender", "high_risk_receiver",
            "fanin_30d", "fanin_intensity_ratio",
            "sent_to_received_ratio_monthly", "back_and_forth_transfers",
            "circular_transaction_count", "Is_laundering"]

# Configuration
features = opt_features[:-1]
target = opt_features[-1]
use_gpu = False             # set True if you have GPU and xgboost built with GPU support
early_stopping_rounds = 50
num_boost_round = 500
verbose_eval = False        # set to integer for logging every k rounds

# Convert Polars -> NumPy once (casting to float32)
# Handle missing values and simple categorical encoding if needed.
def polars_to_numpy_for_xgb(df_pl: pl.DataFrame, features, target):
    # Ensure numeric features; cast numeric-like to Float32
    X_df = df_pl.select([pl.col(c).cast(pl.Float32).alias(c) for c in features])
    y_arr = df_pl.select(pl.col(target)).to_numpy().ravel()
    X_arr = X_df.to_numpy()
    return X_arr, y_arr

X_train, y_train = polars_to_numpy_for_xgb(df_train, features, target)
X_val, y_val = polars_to_numpy_for_xgb(df_val, features, target)

# Build DMatrix once per dataset (faster than rebuilding inside loop)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
watchlist = [(dtrain, "train"), (dval, "eval")]

param_grid = {
    'max_depth': [3, 5, 10],
    'learning_rate': [0.1, 0.05, 0.01],
    'subsample': [0.3, 0.5, 1]
}

# Prediction
def predict_with_best_iter(bst, dmat):
    # bst.best_iteration is int (0-based). If None, predict full model.
    best_it = getattr(bst, "best_iteration", None)
    if best_it is None:
        return bst.predict(dmat)
    # iteration_range expects (begin, end) with end exclusive
    return bst.predict(dmat, iteration_range=(0, best_it + 1))

import time

# Optimized grid search loop using xgb.train
results = []
start_all = time.time()

for max_depth, lr, subs in itertools.product(
        param_grid['max_depth'],
        param_grid['learning_rate'],
        param_grid['subsample']):

    params = {
        "max_depth": int(max_depth),
        "eta": float(lr),                 # alias for learning_rate in xgb.train params
        "subsample": float(subs),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
        "verbosity": 0,
        "nthread": -1
    }
    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    else:
        params["tree_method"] = "hist"

    t0 = time.time()
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    t1 = time.time()

    # predictions on validation (use best_ntree_limit if early stopping occurred)
    preds_proba = predict_with_best_iter(bst, dval)
    preds = (preds_proba > 0.5).astype(int)

    f1 = f1_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    report = classification_report(y_val, preds, digits=4)

    results.append({
        "max_depth": int(max_depth),
        "learning_rate": float(lr),
        "subsample": float(subs),
        "f1_score": f1,
        "precision_score": prec,
        "recall_score": rec,
        "report": report,
        "model": bst,
        "train_time_s": t1 - t0
    })

total_time = time.time() - start_all

# Convert to pandas DataFrame and sort by f1 (descending)
df_results = pd.DataFrame([{k:v for k,v in r.items() if k != "model"} for r in results])
df_results = df_results.sort_values("f1_score", ascending=False).reset_index(drop=True)

# Best model and params
best = results[df_results.index[0]]
best_params = {"max_depth": best["max_depth"], "learning_rate": best["learning_rate"], "subsample": best["subsample"]}
best_model = best["model"]

# Reporting
print(f"Grid search finished in {total_time:.1f}s, tried {len(results)} combos")
print("Top 5 combos by F1:")
print(df_results.head(10))
print("\nBest combo:", best_params)
print("\nClassification report for best model:")
print(best["report"])