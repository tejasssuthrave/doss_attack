
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
