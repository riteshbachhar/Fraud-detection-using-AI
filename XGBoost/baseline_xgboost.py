# Import libaries
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, precision_recall_curve
import pandas as pd
import polars as pl
from datetime import timedelta
import os

opt_features = ["Amount", "year", "day_of_month", "day_of_week", "hour", "minute", "second",
            "currency_mismatch", "high_risk_sender", "high_risk_receiver",
            "fanin_30d", "fanin_intensity_ratio",
            "sent_to_received_ratio_monthly", "back_and_forth_transfers",
            "circular_transaction_count", "Is_laundering"]

# Configuration
features = opt_features[:-1]
target = opt_features[-1]
early_stopping_rounds = 50
num_boost_round = 500
verbose_eval = False        

# Convert Polars -> NumPy once (casting to float32)
# Handle missing values and simple categorical encoding if needed.
def polars_to_numpy_for_xgb(df_pl: pl.DataFrame, features, target):
    # Ensure numeric features; cast numeric-like to Float32
    X_df = df_pl.select([pl.col(c).cast(pl.Float32).alias(c) for c in features])
    y_arr = df_pl.select(pl.col(target)).to_numpy().ravel()
    X_arr = X_df.to_numpy()
    return X_arr, y_arr

X_train, y_train = polars_to_numpy_for_xgb(df_train, features, target)
X_test, y_test = polars_to_numpy_for_xgb(df_test, features, target)

# Build DMatrix once per dataset (faster than rebuilding inside loop)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
watchlist = [(dtrain, "train"), (dtest, "eval")]

params = {
        "max_depth": 10, 
        "eta": 0.01, 
        "subsample": 1,             
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
        "verbosity": 0,
        "nthread": -1
    }

final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        verbose_eval=verbose_eval
    )

# Confusion Matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'High-Risk'],
            yticklabels=['Normal', 'High-Risk'])
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, preds_proba)
test_auc = auc(recall, precision)
avg_prec = average_precision_score(y_test, preds_proba)


plt.figure(figsize=(10, 6))
plt.plot(recall, precision, linewidth=2, label=f'Average Precision = {avg_prec:.4f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# True Positve Rate - False Positive Rate
fpr, tpr, thresholds = roc_curve(y_test, preds_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC-ROC = {roc_auc:.4f}')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


'''
    SHAP analysis
'''
import shap

# Subsample test data
X_sample = X_test_df.sample(n=100000, random_state=42)

# Use optimized TreeExplainer settings
explainer = shap.TreeExplainer(
    final_model,
    feature_perturbation="tree_path_dependent",
    model_output="raw"
)

# Compute SHAP values only on the sample
shap_values = explainer.shap_values(X_sample)

# Use unified SHAP API (optional, if SHAP ≥0.41)
explainer = shap.Explainer(final_model, X_sample)
shap_values = explainer(X_sample)

# Visualize summary plot
shap.summary_plot(shap_values, X_sample)

# Visualize one row
shap.plots.waterfall(shap_values[0])

# Back & forth transfers
back_and_forth_index = X_sample.columns.get_loc("back_and_forth_transfers")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["back_and_forth_transfers"], shap_values[:, back_and_forth_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("back_and_forth_transfers", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs. back_and_forth_transfers", fontsize=14)
plt.colorbar(label="Amount")
plt.show()


# Fan-in intensity ratio
fanin_intensity_ratio_index = X_sample.columns.get_loc("fanin_intensity_ratio")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["fanin_intensity_ratio"], shap_values[:, fanin_intensity_ratio_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("fanin_intensity_ratio", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs fanin_intensity_ratio", fontsize=14)
plt.colorbar(label="Amount")
plt.show()


# Sent-to-received ratio
sent_to_received_ratio_monthly_index = X_sample.columns.get_loc("sent_to_received_ratio_monthly")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["sent_to_received_ratio_monthly"], shap_values[:, sent_to_received_ratio_monthly_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("sent_to_received_ratio_monthly", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs sent_to_received_ratio_monthly", fontsize=14)
plt.colorbar(label="Amount")
plt.show()


# currency mismatch
currency_mismatch_index = X_sample.columns.get_loc("currency_mismatch")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["currency_mismatch"], shap_values[:, currency_mismatch_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("currency_mismatch", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs currency_mismatch", fontsize=14)
plt.colorbar(label="Amount")
plt.show()


# circular transaction count
circular_transaction_count_index = X_sample.columns.get_loc("circular_transaction_count")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["circular_transaction_count"], shap_values[:, circular_transaction_count_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("circular_transaction_count", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs circular_transaction_count", fontsize=14)
plt.colorbar(label="Amount")
plt.show()


# High risk receiver
high_risk_receiver_index = X_sample.columns.get_loc("high_risk_receiver")
plt.figure(figsize=(15, 10))
plt.scatter(X_sample["high_risk_receiver"], shap_values[:, high_risk_receiver_index].values, c=X_sample["Amount"], cmap="viridis")
plt.xlabel("high_risk_receiver", fontsize=12)
plt.ylabel("SHAP value", fontsize=12)
plt.title("SHAP value vs high_risk_receiver", fontsize=14)
plt.colorbar(label="Amount")
plt.show()