#  Hit & Bounce Detection from Ball Trajectories

This project focuses on detecting **ball hits** and **ball bounces** from 2D ball trajectories extracted from video data.  
The problem is addressed using **two complementary approaches**:

- **Unsupervised detection**, based on physical reasoning and signal analysis  
- **Supervised learning**, leveraging labeled data to learn temporal dynamics  

Both approaches aim to infer discrete events from continuous motion and to enrich each input JSON file with a new field:

**`pred_action ∈ {air, hit, bounce}`**

---

##  Problem Overview

Each input file (`ball_data_*.json`) contains frame-level information:

- `(x, y)` pixel coordinates of the ball  
- a `visible` flag indicating whether the ball is detected  
- a ground-truth label `action ∈ {air, hit, bounce}`  

Only the **2D trajectory** is available; no 3D reconstruction or direct video processing is assumed.

The objective is to:

- analyze the **temporal structure** of the trajectory  
- detect **hits** and **bounces** reliably  
- assign an action label to each visible frame  

---

##  Methodology

### 1. Unsupervised Detection (Physics-Based)

The unsupervised approach relies on the physical interpretation of ball motion.

#### Physical intuition

- A **hit** corresponds to a sudden change in velocity direction, especially in the horizontal component  
- A **bounce** corresponds to a sharp change in vertical motion while preserving the overall direction  
- Between events, the motion is smooth and approximately ballistic  

#### Feature analysis

From the raw positions `(x(t), y(t))`, we compute:

- first-order differences: `dx(t)`, `dy(t)` (discrete velocities)  
- second-order differences: `ddx(t)`, `ddy(t)` (discrete accelerations)  
- sign changes and magnitude variations  

These quantities reveal kinematic discontinuities associated with physical events.

#### Event detection

Hits and bounces are inferred by:

- detecting abrupt sign changes in slopes  
- identifying vertical acceleration spikes  
- applying adaptive thresholds estimated from non-event (`air`) frames  

This approach provides:

- an interpretable physical baseline  
- simple and robust heuristics  
- no reliance on labeled data  

---

### 2. Supervised Detection (Learning-Based)

The supervised approach formulates the problem as a **frame-wise classification task** using labeled data.

#### Feature engineering

The same physically motivated features are reused to ensure consistency:

- velocities: `dx`, `dy`  
- accelerations: `ddx`, `ddy`  
- absolute values and norms  
- sign-change indicators  

This bridges the gap between physical reasoning and data-driven learning.

---

### 3. Supervised Models

Several supervised models are trained and compared:

| Model | Type | Temporal Modeling |
|------|------|------------------|
| Random Forest | Tree-based | via engineered features |
| XGBoost | Gradient Boosting | via engineered features |
| MLP | Neural Network | implicit |

- **Tabular models** (RF, XGBoost, MLP) operate on frame-wise feature vectors  
- **Sequence models** (CNN, HMM) explicitly model temporal dependencies  

---

### 4. Training Strategy

- **GroupKFold cross-validation** is used, where each group corresponds to one rally (one JSON file)  
- This prevents temporal leakage between training and validation  
- Hyperparameters are optimized using **GridSearchCV**  

---

### 5. Model Selection Criterion

The final model is selected based on:

- **F1-score on `hit` and `bounce` only**  
- the `air` class is ignored during selection due to strong class imbalance  

This ensures the model focuses on the rare but meaningful events.

---

## 📁 Project Structure and File Description

```text
project/
│
├── main.py
├── model_training.py
├── feature_construction.py
├── computation_bounce_threshold.py
│
├── per_point_v2/
│   ├── ball_data_1.json
│   ├── ball_data_2.json
│   └── ...
│
├── models/
│   ├── rf_model.joblib
│   ├── xgb_model.joblib
│   ├── mlp_model.joblib
│   ├── best_rf_features.pkl
│   ├── best_xgb_features.pkl
│   ├── best_mlp_features.pkl
│   ├── stack.joblib
│   └── scaler.joblib
│
└── README.md
