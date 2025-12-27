import os
import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

from feature_construction import df_per_type


# ============================================================
# Feature selector
# ============================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices=None):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_indices is None:
            return X
        return X[:, self.feature_indices]


# ============================================================
# Feature ranking utilities
# ============================================================

def rank_features_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1]


def rank_features_xgb(X, y):
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X, y)
    return np.argsort(xgb.feature_importances_)[::-1]


def rank_features_mlp(X, y):
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42
        ))
    ])

    mlp.fit(X, y)

    perm = permutation_importance(
        mlp,
        X,
        y,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    return np.argsort(perm.importances_mean)[::-1]


def make_feature_subsets(sorted_idx):
    return [
        list(sorted_idx[:5]),
        list(sorted_idx[:8]),
        list(sorted_idx[:12]),
        list(sorted_idx)
    ]


# ============================================================
# Main training function
# ============================================================

def train_select_models(FOLDER, FEATURES, LABEL_MAP):

    # =========================
    # Load data
    # =========================

    X_seq, y_seq, groups = [], [], []

    json_ids = sorted([
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(FOLDER)
        if f.startswith("ball_data_") and f.endswith(".json")
    ])

    for pid in json_ids:
        df = df_per_type(pid)
        if df is None:
            continue

        df_vis = df[df["visible"]].copy()
        df_vis = df_vis.dropna(subset=FEATURES + ["action"])

        if len(df_vis) < 6:
            continue

        X_seq.append(df_vis[FEATURES].values)
        y_seq.append(df_vis["action"].map(LABEL_MAP).values)
        groups.append(pid)

    X = np.vstack(X_seq)
    y = np.hstack(y_seq)
    group_flat = np.repeat(groups, [len(x) for x in X_seq])

    gkf = GroupKFold(n_splits=5)

    # =========================
    # Feature ranking per model
    # =========================

    print("\n===== Feature ranking per model =====")

    rf_rank = rank_features_rf(X, y)
    xgb_rank = rank_features_xgb(X, y)
    mlp_rank = rank_features_mlp(X, y)

    rf_subsets = make_feature_subsets(rf_rank)
    xgb_subsets = make_feature_subsets(xgb_rank)
    mlp_subsets = make_feature_subsets(mlp_rank)

    # =========================
    # RANDOM FOREST
    # =========================

    print("\n===== RANDOM FOREST =====")

    pipe_rf = Pipeline([
        ("selector", FeatureSelector()),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ])

    rf_grid = {
        "selector__feature_indices": rf_subsets,
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 15],
        "model__min_samples_leaf": [1, 5],
    }

    rf_search = GridSearchCV(
        pipe_rf,
        rf_grid,
        cv=list(gkf.split(X, y, group_flat)),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X, y)
    best_rf = rf_search.best_estimator_

    # =========================
    # XGBOOST
    # =========================

    print("\n===== XGBOOST =====")

    pipe_xgb = Pipeline([
        ("selector", FeatureSelector()),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    xgb_grid = {
        "selector__feature_indices": xgb_subsets,
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.03, 0.05],
    }

    xgb_search = GridSearchCV(
        pipe_xgb,
        xgb_grid,
        cv=list(gkf.split(X, y, group_flat)),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    xgb_search.fit(X, y)
    best_xgb = xgb_search.best_estimator_

    # =========================
    # MLP
    # =========================

    print("\n===== MLP =====")

    pipe_mlp = Pipeline([
        ("selector", FeatureSelector()),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(max_iter=300, random_state=42))
    ])

    mlp_grid = {
        "selector__feature_indices": mlp_subsets,
        "model__hidden_layer_sizes": [(64,), (64, 32)],
        "model__alpha": [1e-4, 1e-3],
    }

    mlp_search = GridSearchCV(
        pipe_mlp,
        mlp_grid,
        cv=list(gkf.split(X, y, group_flat)),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    mlp_search.fit(X, y)
    best_mlp = mlp_search.best_estimator_

    # =========================
    # STACKING
    # =========================

    print("\n===== STACKING =====")

    stack = StackingClassifier(
        estimators=[
            ("rf", best_rf),
            ("xgb", best_xgb),
            ("mlp", best_mlp)
        ],
        final_estimator=LogisticRegression(max_iter=500),
        n_jobs=-1
    )

    stack_scores = cross_val_score(
        stack,
        X,
        y,
        cv=gkf.split(X, y, group_flat),
        scoring="f1_macro",
        n_jobs=-1
    )

    print(f"Stacking F1_macro: {stack_scores.mean():.4f} ± {stack_scores.std():.4f}")

    stack.fit(X, y)
    
    best_rf_features  = rf_search.best_params_["selector__feature_indices"]
    best_xgb_features = xgb_search.best_params_["selector__feature_indices"]
    best_mlp_features = mlp_search.best_params_["selector__feature_indices"]


    # =========================
    # Return
    # =========================

    return {
        "rf": (best_rf, best_rf_features),
        "xgb": (best_xgb, best_xgb_features),
        "mlp": (best_mlp, best_mlp_features),
        "stack": stack
    }

