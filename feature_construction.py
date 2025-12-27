import json
import pandas as pd
import numpy as np

def df_per_type(i, window=3):
    try:
        with open(f"per_point_v2/ball_data_{i}.json", "r") as f:
            data_1 = json.load(f)

        df_1 = (
            pd.DataFrame.from_dict(data_1, orient="index")
            .reset_index()
            .rename(columns={"index": "frame"})
        )
        df_1 = df_1.sort_values("frame").reset_index(drop=True)

        # =========================
        # Multi-point backward / forward deltas
        # =========================

        # backward means
        x_back = (
            df_1["x"]
            .rolling(window=window, min_periods=window)
            .mean()
            .shift(1)
        )
        y_back = (
            df_1["y"]
            .rolling(window=window, min_periods=window)
            .mean()
            .shift(1)
        )

        # forward means
        x_forward = (
            df_1["x"]
            .shift(-1)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        y_forward = (
            df_1["y"]
            .shift(-1)
            .rolling(window=window, min_periods=window)
            .mean()
        )

        # deltas
        df_1["dx-"] = df_1["x"] - x_back
        df_1["dx+"] = x_forward - df_1["x"]
        df_1["dy-"] = df_1["y"] - y_back
        df_1["dy+"] = y_forward - df_1["y"]

        # magnitudes
        df_1["|dx-|"] = df_1["dx-"].abs()
        df_1["|dx+|"] = df_1["dx+"].abs()
        df_1["|dy-|"] = df_1["dy-"].abs()
        df_1["|dy+|"] = df_1["dy+"].abs()

        # signs
        df_1["sign_dx"] = np.sign(df_1["dx-"] * df_1["dx+"])
        df_1["sign_dy"] = np.sign(df_1["dy-"] * df_1["dy+"])
        df_1["sign_dx_dy"] = np.sign(df_1["dx-"] * df_1["dy-"])

        # second-order differences
        df_1["ddx"] = df_1["dx-"] - df_1["dx+"]
        df_1["ddy"] = df_1["dy-"] - df_1["dy+"]

        # speed
        df_1["v"] = np.sqrt(df_1["dx+"]**2 + df_1["dy+"]**2)

        # temporal shifts (kept for ML)
        df_1["dx--"] = df_1["dx-"].shift(1)
        df_1["dy--"] = df_1["dy-"].shift(1)
        df_1["dx++"] = df_1["dx+"].shift(-1)
        df_1["dy++"] = df_1["dy+"].shift(-1)

        df_1["v-"] = df_1["v"].shift(1)
        df_1["v--"] = df_1["v-"].shift(1)
        df_1["v+"] = df_1["v"].shift(-1)
        df_1["v++"] = df_1["v+"].shift(-1)
        
        min_dx_m = df_1["dx-"][df_1["dx-"] > 0].min()
        min_dx_p = df_1["dx+"][df_1["dx+"] > 0].min()

        df_1["dy-/dx-"] = df_1['dy-']/(df_1['dx-']+min_dx_m/1000)
        df_1["dy+/dx+"] = df_1['dy+']/(df_1['dx+']+min_dx_p/1000)
        
        last_hit_idx = None
        last_bounce_idx = None

        for idx, action in enumerate(df_1.get("action", [])):
            if last_hit_idx is not None:
                df_1.at[idx, "dist_to_last_hit"] = idx - last_hit_idx
            if last_bounce_idx is not None:
                df_1.at[idx, "dist_to_last_bounce"] = idx - last_bounce_idx

            if action == "hit":
                last_hit_idx = idx
                df_1.at[idx, "dist_to_last_hit"] = 0

            if action == "bounce":
                last_bounce_idx = idx
                df_1.at[idx, "dist_to_last_bounce"] = 0

        # Optional: replace NaN by large constant (useful for ML models)
        max_dist = len(df_1)
        df_1["dist_to_last_hit"] = df_1["dist_to_last_hit"].fillna(max_dist)
        df_1["dist_to_last_bounce"] = df_1["dist_to_last_bounce"].fillna(max_dist)


        return df_1
    
    except Exception :
        print("file not found")