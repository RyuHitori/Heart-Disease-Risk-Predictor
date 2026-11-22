# Heart Disease Risk Predictor

BDA700 Health Analytics, Project Milestone 2: a machine learning classification app deployed with Streamlit Community Cloud.

**Live app:** [[https://your-app-name.streamlit.app](https://heart-disease-risk-predictor-srvajrjxhhxopt3cuhprpm.streamlit.app/)](https://heart-disease-risk-predictor-srvajrjxhhxopt3cuhprpm.streamlit.app/)

## Project description
The app predicts whether a patient is **Normal** or at **Risk** of ischemic heart disease from 13 clinical measurements (age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, and others). The user enters the values, clicks **Predict**, and receives the predicted class and the probability of each class.

## Dataset
UCI Heart Disease, Cleveland cohort (303 patients, 13 features). The original target `num` (0 to 4) was converted to a binary target: **0 = Normal**, **1 = Risk** (original levels 1 to 4).

## Model
A scikit-learn `Pipeline` (median imputer, standard scaler, Random Forest classifier) trained on an 80/20 stratified train/test split and evaluated with accuracy, precision, recall, F1-score and a confusion matrix. The trained pipeline is saved with `joblib` as `classification_model.joblib`.

## Repository files
| File | Purpose |
|---|---|
| `classification_app.py` | Streamlit application |
| `classification_model.joblib` | Trained and saved model |
| `requirements.txt` | Packages needed for deployment |
| `README.md` | Project description |

## Run locally
```bash
pip install -r requirements.txt
streamlit run classification_app.py
```

## Disclaimer
This project is for educational purposes only and is not medical advice.
