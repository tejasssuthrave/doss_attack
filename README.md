# DDoS Attack Detection Using Machine Learning

## 📌 Project Overview
This project focuses on detecting Distributed Denial of Service (DDoS) attacks using machine learning techniques. By analyzing network traffic features, the system classifies traffic as either **BENIGN** or **DDoS**. A Random Forest classifier is trained on the CIC-IDS dataset to achieve high detection accuracy.

The goal of this project is to demonstrate how data preprocessing, feature engineering, and supervised learning can be applied to cybersecurity problems.

---

## 📂 Dataset Description
- **Dataset Name:** Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  
- **Source:** CIC-IDS (Canadian Institute for Cybersecurity)
- **Records:** 225,745
- **Features:** 79 network traffic features
- **Target Column:** `Label`  
  - `BENIGN`
  - `DDoS`

### Dataset Characteristics
- No missing values initially
- Presence of infinite values in some flow features
- Class imbalance:
  - DDoS: 128,027
  - BENIGN: 97,718

---

## 🧹 Data Preprocessing Steps
1. **Column Name Cleaning**
   - Removed leading/trailing spaces from column names

2. **Removal of Constant Features**
   - Columns with only one unique value were dropped

3. **Handling Infinite Values**
   - Replaced `inf` and `-inf` with NaN
   - Removed rows containing NaN values

4. **Feature & Label Separation**
   - Features (X): All columns except `Label`
   - Target (y): `Label`

5. **Label Encoding**
   - BENIGN → 0
   - DDoS → 1

6. **Feature Scaling**
   - Applied `StandardScaler` to normalize numerical features

---

## 🧪 Train-Test Split
- **Training Set:** 80%
- **Testing Set:** 20%
- **Random State:** 42
- **Stratified Split:** Ensures class balance

---

## 🤖 Machine Learning Model
### Model Used
- **Algorithm:** Random Forest Classifier
- **Reason for Choice:**
  - Handles high-dimensional data well
  - Robust to noise
  - Reduces overfitting through ensemble learning

### Hyperparameters
- `n_estimators = 200`
- `random_state = 42`
- `n_jobs = -1`

---

## 📊 Model Evaluation Metrics
The model was evaluated using multiple performance metrics:

| Metric | Score |
|------|------|
| Accuracy | 0.9999 |
| Precision | 1.0000 |
| Recall | 0.9999 |
| F1-score | 0.9999 |

### Confusion Matrix
[[19543 1]
[ 1 25604]]

### ROC-AUC
- ROC curve plotted
- Area Under Curve (AUC) ≈ **1.0**

---

## 🔍 Cross Validation
- **Method:** Stratified K-Fold (k=5)
- **Mean Accuracy:** 0.9999
- **Standard Deviation:** 0.0001

This confirms the model’s stability and generalization capability.

---

## 🚨 Data Leakage Check
- Checked for duplicate rows
- Verified no overlap between training and testing data
- No features showed suspicious correlation (>0.9999) with target

✅ No data leakage detected.

---

## ⭐ Feature Importance
Permutation Importance identified the most influential features:

Top features include:
- Destination Port
- Init_Win_bytes_forward
- Init_Win_bytes_backward
- FIN Flag Count
- URG Flag Count

These features are strongly correlated with abnormal traffic behavior.

---

## 🧪 DDoS Detection Demonstration
The trained model was tested on selected BENIGN and DDoS samples.

| Actual Label | Predicted Label |
|-------------|----------------|
| BENIGN | BENIGN |
| DDoS | DDoS |

The model correctly classified all selected instances.

---

## 💾 Model Saving
The trained model and preprocessing artifacts were saved using `joblib`.

### Saved Files
- `ddos_rf_model.pkl` → Trained Random Forest model
- `ddos_preprocessing.pkl` → Scaler, Imputer, Label Encoder, Feature Columns

These files can be reused for real-time or batch inference.

---

## 🛠️ Technologies Used
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook / Google Colab

---

## 🚀 Conclusion
This project successfully demonstrates an end-to-end machine learning pipeline for DDoS attack detection. The Random Forest model achieved near-perfect performance, proving the effectiveness of network flow features for intrusion detection.

Future improvements include:
- Testing on real-time traffic
- Deploying the model using REST APIs
- Evaluating performance on unseen datasets

---

## 📌 Author
**Project Title:** DDoS Attack Detection  
**Course:** Cybersecurity / Machine Learning  
**Status:** Academic Assignment
