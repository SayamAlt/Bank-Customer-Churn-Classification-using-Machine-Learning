# Customer Churn Prediction using Machine Learning in R

## 🚀 Project Overview
Successfully developed a **customer churn classification system** using advanced machine learning algorithms in **R**. This project aims to identify customers who are likely to leave a service based on their demographic, behavioral, and transactional data. The solution includes **end-to-end data processing**, **model building**, and **evaluation pipelines** optimized for real-world deployment.

---

## 🔑 Key Features
- **Exploratory Data Analysis (EDA):** In-depth data visualization using `ggplot2`, `plotly`, and correlation heatmaps.
- **Feature Engineering:** 
  - One-hot encoding of categorical variables.
  - Outlier detection and imputation.
  - Class imbalance handled using **SMOTE**.
- **Feature Scaling:** Min-max normalization for continuous features.
- **Model Building:** Implemented and tuned **20+ ML models**, including:
  - Logistic Regression (Binary & Multinomial)
  - Decision Trees, Random Forest, C5.0
  - Gradient Boosting (GBM), XGBoost
  - Bagging & AdaBoost
  - SVM (Radial Kernel)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Neural Networks (nnet & Keras)
  - Regularization methods (Lasso & Ridge)
- **Hyperparameter Tuning:** Automated model optimization using `caret` with grid search and cross-validation.
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, and confusion matrices.
- **Model Comparison:** Automatic evaluation and ranking of models based on performance metrics.
- **Deployment Ready:** Best-performing model is saved using `saveRDS()` for future use.

---

## 📊 Results
- Achieved **100% accuracy with KNN** and **90%+ accuracy with XGBoost** after hyperparameter tuning.
- Robust pipeline enabling easy retraining and scalability.

---

## ⚙️ Tech Stack
- **Language:** R
- **Libraries:** `caret`, `ggplot2`, `plotly`, `xgboost`, `lightgbm`, `nnet`, `keras`, `e1071`, `rpart`, `randomForest`, `adabag`, `MASS`, `glmnet`, `smotefamily`.

---

## 🏆 Use Cases
- Telecom, Banking, and SaaS industries to identify customers likely to churn.
- Actionable insights for **customer retention strategies**.

---

## 📈 Future Enhancements
- Integration with **Shiny dashboard** for interactive churn visualization.
- Model deployment via **REST API** or containerization with Docker.