import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_construction import df_per_type
from model_training import train_select_models

import joblib

def unsupervised_hit_bounce_detection(json_file_title, T_bounce=35, window=10):
    """
    Assign pred_action = hit / bounce / air
    using multi-point backward and forward deltas.
    """

    import json
    import numpy as np

    with open(f"per_point_v2/{json_file_title}", "r") as f:
        ball_data = json.load(f)

    frames = sorted(ball_data.keys())

    # Initialisation
    for f in frames:
        ball_data[f]["pred_action"] = "air"

    # Frames visibles
    vis_frames = []
    x_vals, y_vals = [], []

    for f in frames:
        d = ball_data[f]
        if d.get("visible") and d.get("x") is not None and d.get("y") is not None:
            vis_frames.append(f)
            x_vals.append(d["x"])
            y_vals.append(d["y"])

    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)

    n = len(x_vals)
    if n < 2 * window + 1:
        return ball_data

    # Parcours frame par frame (éviter les bords)
    for i in range(window, n - window):

        frame = vis_frames[i]

        # ---- deltas arrière lissés ----
        x_back_mean = np.mean(x_vals[i - window:i])
        y_back_mean = np.mean(y_vals[i - window:i])

        dx_minus = x_vals[i] - x_back_mean
        dy_minus = y_vals[i] - y_back_mean

        # ---- deltas avant lissés ----
        x_forward_mean = np.mean(x_vals[i + 1:i + 1 + window])
        y_forward_mean = np.mean(y_vals[i + 1:i + 1 + window])

        dx_plus = x_forward_mean - x_vals[i]
        dy_plus = y_forward_mean - y_vals[i]

        # ---- estimation saut vertical (bounce) ----
        ddy = dy_plus - dy_minus

        # -------- HIT --------
        if dx_minus * dx_plus < 0 and dy_minus * dy_plus < 0:
            ball_data[frame]["pred_action"] = "hit"

        # -------- BOUNCE --------
        elif (
            dx_minus * dx_plus > 0 and
            dy_minus * dy_plus < 0 and
            abs(ddy) > T_bounce
        ):
            ball_data[frame]["pred_action"] = "bounce"

        # -------- AIR --------
        else:
            ball_data[frame]["pred_action"] = "air"

    return ball_data


FOLDER = "per_point_v2"

FEATURES = [
    'dx-', '|dx-|', 'dx+', '|dx+|', 'dx--', 'dx++',
    'dy-', '|dy-|', 'dy+', '|dy+|', 'dy--', 'dy++',
    'ddx', 'ddy',
    'sign_dx', 'sign_dy', 'sign_dx_dy',
    'v', 'v-', 'v+', 'v--', 'v++',
    "dy-/dx-", "dy+/dx+",
    "dist_to_last_hit", "dist_to_last_bounce",
]

LABEL_MAP = {"air": 0, "hit": 1, "bounce": 2}


"""candidates_dict = train_select_models(FOLDER, FEATURES, LABEL_MAP)

best_rf, best_rf_features = candidates_dict["rf"]
best_xgb, best_xgb_features = candidates_dict["xgb"]
best_mlp, best_mlp_features = candidates_dict["mlp"]
stack = candidates_dict["stack"]"""

scaler = StandardScaler()

best_rf = joblib.load("models/rf_model.joblib")
best_xgb = joblib.load("models/xgb_model.joblib")
best_mlp = joblib.load("models/mlp_model.joblib")
stack = joblib.load("models/stack.joblib")
best_rf_features = joblib.load("models/best_rf_features.pkl")
best_xgb_features = joblib.load("models/best_xgb_features.pkl")
best_mlp_features = joblib.load("models/best_mlp_features.pkl")


def supervized_hit_bounce_detection(json_path):
    """
    - Applique le modèle final (stacking)
    - enrichit le JSON avec pred_action
    - retourne le JSON enrichi
    """

    # =========================
    # 1. Charger et feature-engineer le JSON
    # =========================

    point_id = int(json_path.split("_")[-1].split(".")[0])
    out = df_per_type(point_id)

    if out is None:
        raise ValueError("JSON invalide")

    df_all = out
    df_vis = df_all[df_all["visible"]].copy()
    df_vis = df_vis.fillna(0)

    X = df_vis[FEATURES].values

    # =========================
    # 2. Prédiction avec le modèle final (STACK)
    # =========================

    y_pred = stack.predict(X)

    # =========================
    # 3. Enrichir le JSON
    # =========================

    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    df_vis["pred_action"] = [inv_label_map[p] for p in y_pred]

    # réinjecter dans le JSON original

    df_all = df_all.merge(
        df_vis[["frame", "pred_action"]],
        on="frame",
        how="left"
    )

    df_all["pred_action"] = df_all["pred_action"].fillna("air")
    df_all = df_all[["frame", "x", "y", "visible", "action", "pred_action"]]

    json_data = (
        df_all.set_index("frame")
        .to_dict(orient="index")
    )

    # =========================
    # 4. Retour
    # =========================

    return json_data



if __name__ =="__main__" :
    """os.makedirs("models", exist_ok=True)

    joblib.dump(best_rf, "models/rf_model.joblib")
    joblib.dump(best_xgb, "models/xgb_model.joblib")
    joblib.dump(best_mlp, "models/mlp_model.joblib")
    joblib.dump(stack, "models/stack.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(best_rf_features, "models/best_rf_features.pkl")
    joblib.dump(best_xgb_features, "models/best_xgb_features.pkl")
    joblib.dump(best_mlp_features, "models/best_mlp_features.pkl")
"""
    
    """json_1_unsupervised = unsupervised_hit_bounce_detection(r"ball_data_1.json")
    json_1_supervised = supervized_hit_bounce_detection(r"ball_data_1.json")
    print("UNSUPERVISED TYPE =", type(json_1_unsupervised))
    print("SUPERVISED TYPE =", type(json_1_supervised))"""


    """df_1_unsupervised = (pd.DataFrame.from_dict(json_1_unsupervised, orient="index")
            .reset_index()
            .rename(columns={"index": "frame"}))
    df_1_supervised = (pd.DataFrame.from_dict(json_1_supervised, orient="index")
            .reset_index()
            .rename(columns={"index": "frame"}))
    df_hb_uns = df_1_unsupervised[df_1_unsupervised["action"].isin(["hit", "bounce"])]
    df_hb_s = df_1_supervised[df_1_unsupervised["action"].isin(["hit", "bounce"])]"""
    
    dfs = []

    for i in range(50):
        try :
            json_data = unsupervised_hit_bounce_detection(f"ball_data_{i}.json")
            
            df = (pd.DataFrame.from_dict(json_data, orient="index")
                    .reset_index()
                    .rename(columns={"index": "frame"}))
        except Exception :
            continue
        
        dfs.append(df)

    df_final_unsupervised = pd.concat(dfs, ignore_index=True)
    
    for i in range(50):
        try :
            json_data = supervized_hit_bounce_detection(f"ball_data_{i}.json")
            
            df = (pd.DataFrame.from_dict(json_data, orient="index")
                    .reset_index()
                    .rename(columns={"index": "frame"}))
        except Exception :
            continue
        
        dfs.append(df)

    df_final_supervised = pd.concat(dfs, ignore_index=True)
    
    df_hb_uns = df_final_unsupervised[df_final_unsupervised["action"].isin(["hit", "bounce"])]
    df_hb_s = df_final_supervised[df_final_supervised["action"].isin(["hit", "bounce"])]


    
    print("------- Unsupervised -----------")
    #print(df_final_unsupervised.head())
    print(classification_report(df_final_unsupervised["action"], df_final_unsupervised["pred_action"]))
    accuracy_hb = (df_hb_uns["action"] == df_hb_uns["pred_action"]).mean()
    print("Accuracy hit/bounce :", accuracy_hb)


    print("------- Supervised -----------")
    #print(df_final_supervised.head())
    print(classification_report(df_final_supervised["action"], df_final_supervised["pred_action"]))
    accuracy_hb = (df_hb_s["action"] == df_hb_s["pred_action"]).mean()
    print("Accuracy hit/bounce :", accuracy_hb)

