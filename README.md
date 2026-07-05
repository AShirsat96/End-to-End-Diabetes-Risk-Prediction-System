# End-to-End Diabetes Risk Prediction System

An end-to-end machine learning system for predicting diabetes risk using **Apache Spark**, **PySpark ML**, **AWS SageMaker Serverless Inference**, and an interactive **Dash** web application.

The project demonstrates the complete machine learning lifecycle—from distributed data processing and model development to cloud deployment and a live clinical decision support dashboard.

---

# 🌐 Live Demo

**Interactive Dashboard**

https://diabetes-risk-dashboard.onrender.com/

The deployed dashboard allows users to:

- Predict diabetes risk using a trained machine learning model
- Visualize prediction confidence through an interactive risk gauge
- Receive personalized clinical recommendations
- Explore predictions through a user-friendly web interface

---

# Project Overview

Diabetes is one of the world's fastest-growing chronic diseases, making early risk assessment increasingly important for improving patient outcomes.

This project presents an end-to-end machine learning pipeline for diabetes risk prediction using Apache Spark for scalable data processing, PySpark ML for model development, AWS SageMaker Serverless for deployment, and a Dash dashboard for interactive inference.

Rather than focusing solely on model accuracy, this project demonstrates how a machine learning solution can be engineered into a deployable application capable of serving real-time predictions.

---

# Business Problem

Healthcare providers often require fast and accurate risk assessment tools to identify patients who may require further screening for diabetes.

Traditional manual assessments can be time-consuming and difficult to scale across large patient populations.

This project explores how distributed machine learning and cloud deployment technologies can be combined to build an intelligent clinical decision support system capable of generating real-time diabetes risk predictions.

---

# System Architecture

<img width="582" height="423" alt="SytemArchitecture" src="https://github.com/user-attachments/assets/79f18127-abf9-45c0-a8b5-6f1aee67df35" />


The system consists of the following components:

- Apache Spark for distributed data processing
- Feature engineering and preprocessing pipeline
- Machine learning model training
- AWS SageMaker Serverless deployment
- Dash dashboard for interactive prediction
- Live deployment using Render

---

# Machine Learning Pipeline

The project follows a complete end-to-end machine learning workflow.

### Data Processing

- Apache Spark DataFrames
- Data cleaning
- Missing value handling
- Feature engineering
- Feature correlation analysis

### Model Development

Multiple machine learning algorithms were evaluated including:

- Logistic Regression
- Random Forest
- Gradient Boosted Trees (GBT)
- XGBoost
- CatBoost

### Model Optimization

- Cross Validation
- Hyperparameter tuning
- Class imbalance handling
- Performance comparison

### Model Evaluation

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# Interactive Dashboard

The deployed Dash application provides an intuitive interface for diabetes risk prediction.

Features include:

- Patient data input
- Real-time prediction
- Interactive risk gauge
- Personalized clinical recommendations
- Cloud-hosted inference

<img width="945" height="2048" alt="WhatsApp Image 2025-12-14 at 22 10 35_ce4de3df" src="https://github.com/user-attachments/assets/5c4bae03-2456-48a0-866a-15541ef97182" />

<img width="945" height="2048" alt="WhatsApp Image 2025-12-14 at 22 10 35_409a2724" src="https://github.com/user-attachments/assets/7d8be462-c605-473d-8385-6d3c16b7f70a" />


---

# Model Evaluation

Model performance was evaluated using multiple classification metrics to balance predictive accuracy with clinical interpretability.

The project also explored threshold selection to better understand the trade-off between recall and precision when identifying high-risk patients.

<img width="598" height="372" alt="RecallVsPrecisionTradeoffandRiskLevelThreshold" src="https://github.com/user-attachments/assets/3df71457-9fa5-4fdb-82ea-595c3633314f" />


---

# Technology Stack

## Machine Learning

- Apache Spark
- PySpark MLlib
- Scikit-learn
- XGBoost
- CatBoost

## Cloud

- AWS SageMaker Serverless
- Amazon S3
- Boto3

## Dashboard

- Dash
- Plotly

## Programming Language

- Python

## Deployment

- Render

---

# Repository Structure

```text
End-to-End-Diabetes-Risk-Prediction-System/

├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── End_to_End_Diabetes_Risk_Prediction.ipynb
├── deployment_script.py
├── app.py
├── render.yaml
│
└── images/
    ├── system_architecture.png
    ├── dashboard.png
    └── risk_thresholds.png
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/AShirsat96/End-to-End-Diabetes-Risk-Prediction-System.git
cd End-to-End-Diabetes-Risk-Prediction-System
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Launch the dashboard locally.

```bash
python app.py
```

---

# Skills Demonstrated

This project demonstrates practical experience with:

- Distributed data processing using Apache Spark
- Machine learning pipeline development
- Feature engineering
- Model evaluation and comparison
- Hyperparameter tuning
- AWS SageMaker deployment
- Serverless inference
- Dash dashboard development
- End-to-end machine learning engineering

---

# Project Context

This project was completed as part of a university Cloud Computing group project.

This repository contains the final project implementation, including the Apache Spark machine learning pipeline, deployment scripts, and interactive dashboard application.

---

# Future Improvements

Potential future enhancements include:

- Automated model retraining
- CI/CD for model deployment
- Docker containerization
- Kubernetes deployment
- Model monitoring
- Explainable AI (SHAP/LIME)
- User authentication
- Batch prediction support

---

# About the Author

**Aniket Shirsat**

AI Engineer | Data Scientist | Generative AI

- GitHub: https://github.com/AShirsat96
- LinkedIn: https://www.linkedin.com/in/aniketshirsatsg/
- Portfolio: https://aniketdshirsat.com

---

# License

This project is licensed under the MIT License.
