# 📡 TeleChurn AI

> **Customer Churn Prediction System using Machine Learning and DevOps**

TeleChurn AI is an end-to-end Machine Learning and DevOps project developed to predict customer churn in the telecommunications industry. The project compares two supervised learning algorithms—**Logistic Regression** and **Decision Tree**—through an interactive Streamlit web application.

The project also demonstrates modern DevOps practices including Git version control, Docker containerization, Continuous Integration (CI) using GitHub Actions, automated testing, and cloud deployment.

---

## 🌐 Live Demo

**Live Application:** *https://telechurn-ai.streamlit.app/*

---

## 📷 Project Preview

<img width="1918" height="834" alt="image" src="https://github.com/user-attachments/assets/ffd2108e-83db-4591-93de-9fe216dd9827" />

---

## 🎯 Objectives

* Predict customer churn using Machine Learning.
* Compare Logistic Regression and Decision Tree performance.
* Build an interactive Streamlit dashboard.
* Dockerize the application.
* Automate testing using GitHub Actions.
* Deploy the project on Streamlit Community Cloud.

---

## ✨ Features

* Customer churn prediction
* Two machine learning models
* Model comparison
* Confidence score
* Probability distribution chart
* Interactive Streamlit interface
* Docker support
* CI Pipeline with GitHub Actions
* Live cloud deployment

---

## 🧰 Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn

### Frontend

* Streamlit

### DevOps

* Git
* GitHub
* Docker
* GitHub Actions

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Label Encoding
6. Standard Scaling
7. Train/Test Split
8. Logistic Regression
9. Decision Tree
10. Model Evaluation
11. Model Export (.pkl)
12. Streamlit Deployment

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 77.33%   |
| Decision Tree       | 72.35%   |

---

## 📂 Project Structure

```text
TeleChurn-AI/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── scaler.pkl
│   ├── contract_encoder.pkl
│   ├── internet_encoder.pkl
│   ├── security_encoder.pkl
│   ├── support_encoder.pkl
│   └── payment_encoder.pkl
│
├── notebook/
│   └── Customer_Churn_Prediction.ipynb
│
├── tests/
│   └── test_app.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Run Using Docker

Build Docker image

```bash
docker build -t telechurn-ai .
```

Run Docker container

```bash
docker run -p 8501:8501 telechurn-ai
```

---

## 💻 Run Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## ⚙️ Continuous Integration

GitHub Actions automatically performs:

* Dependency installation
* Unit testing
* Build verification

Every push to the **main** branch triggers the CI pipeline.

---

## 📌 Future Improvements

* Random Forest implementation
* XGBoost model
* Hyperparameter tuning
* SHAP explainability
* Database integration
* User authentication
* Cloud deployment using Azure/AWS

---

## 👨‍💻 Developer

**Hasnain Ahmed Ansari (Team Lead), Allah Bux, Hoorain Mazhar, M. Usman Faisal, & M. Bilal**

Bachelor of Computer Science - Muhammad Ali Jinnah University

---

## 📄 License

This project is developed for educational and portfolio purposes.
