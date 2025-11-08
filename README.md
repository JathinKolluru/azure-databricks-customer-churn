# 🧠 Customer Churn Prediction on Azure Databricks

### 📘 Overview
This project predicts customer churn using scalable machine learning workflows built on **Azure Databricks**.  
It demonstrates data preprocessing, model training, and experiment tracking with **MLflow**, integrating with **Azure Machine Learning** for deployment and real-time scoring.

---

### ⚙️ Tools & Technologies
- **Programming:** Python (pandas, scikit-learn, matplotlib)
- **Cloud Platform:** Azure Databricks
- **Experiment Tracking:** MLflow
- **Deployment:** Azure Machine Learning
- **Version Control:** Git & GitHub

---

### 🎯 Project Highlights
- Processed and modeled **10,000+ CRM (Customer Relationship Management) records**, including data cleaning, feature encoding, and scaling.
- Built a reproducible ML pipeline comparing **Logistic Regression, Random Forest, and Gradient Boosting**.
- Achieved a **14% improvement in prediction accuracy** over the baseline Logistic Regression model.
- Integrated trained model with **Azure ML** for real-time inference, **reducing manual churn analysis effort by ~40%**.
- Tracked all experiments, metrics (Accuracy, Precision, Recall, ROC-AUC), and artifacts using **MLflow** within Databricks.

---

### 🧪 Model Performance Summary
| Model | Accuracy | ROC-AUC | Key Insight |
|--------|-----------|---------|--------------|
| Logistic Regression | 0.84 | 0.90 | Baseline |
| Random Forest | 0.93 | 0.97 | Improved performance |
| Gradient Boosting | **0.95** | **1.00** | Best performing model |

---

### 🧰 MLflow Experiment Tracking
- Each training run automatically logs:
  - Parameters (model type, hyperparameters)
  - Metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Artifacts (confusion matrix plots, trained model)
- Experiments are stored under:  
  `/Users/jkolluru@charlotte.edu/Churn_Experiments`

---

### 📂 Folder Structure
azure-databricks-customer-churn/
│
├── data/                        # Dataset (customer_churn_large_ds.csv)
├── src/                         # Training scripts
│   └── churn_training.py
├── reports/                     # Model artifacts and plots
│   ├── churn_model.pkl
│   └── figures/
│       └── confusion_matrix.png
├── notebooks/                   # Databricks notebook exports
│   └── Customer_Churn_Prediction_Databricks.html
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

---

### 🚀 Key Learnings
- Built scalable ML pipelines using **Azure Databricks** clusters.
- Implemented **MLflow** for robust experiment management and reproducibility.
- Integrated **cloud deployment workflows** via Azure ML.
- Improved project collaboration and version tracking through **GitHub**.

---

### 👨‍💻 Author
**Jathin Kolluru**  
Master’s in Computer Science (Data Science Concentration), UNC Charlotte  

🔗 [LinkedIn](https://linkedin.com/in/jathinkolluru)  
💻 [GitHub](https://github.com/jathinkolluru)

---

### 📈 Future Enhancements
- Add **k-fold cross-validation** for performance stability.
- Implement **SHAP explainability plots** for feature importance.
- Automate **CI/CD deployment** with Azure DevOps.

---