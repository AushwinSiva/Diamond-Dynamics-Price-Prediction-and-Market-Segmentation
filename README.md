💎 Diamond Dynamics — Price Prediction & Market Segmentation

A complete end-to-end data science project that predicts diamond prices and identifies market segments using machine learning, clustering, and a Streamlit web application.

📌 Project Overview

This project builds a robust ML system to understand the diamond market by:
Predicting diamond prices based on physical and categorical attributes
Segmenting diamonds into market categories using clustering
Deploying the prediction system through an interactive Streamlit application
The solution combines Exploratory Data Analysis, Feature Engineering, Regression Modeling, ANN, Clustering, and PCA Visualization.

📊 Key Features
🔹 1. Exploratory Data Analysis (EDA)

Distribution analysis for numerical features
Category-wise price variation (cut, color, clarity)
Correlation heatmaps
Outlier analysis using IQR
Scatterplots and pairplots for in-depth insights

🔹 2. Feature Engineering & Preprocessing

Handling missing/invalid values
Log transformations for skewness
New features: volume, dimension_ratio, price_per_carat
Ordinal encoding for categorical columns
Feature importance + Recursive Feature Selection

🔹 3. Price Prediction Models

Models implemented and evaluated using MAE, MSE, RMSE, and R²:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
KNN Regressor
XGBoost Regressor
ANN (TensorFlow/Keras)
Random Forest achieved the best balance of accuracy, performance, and deployment compatibility, and was selected for the final app.

🔹 4. Market Segmentation (Clustering)

K-Means clustering (optimal K via Elbow + Silhouette Score)
PCA for dimensionality reduction (2D visualization)
Cluster naming based on averages of price, carat, and dimensions:
Premium Heavy Diamonds
Mid-range Balanced Diamonds
Affordable Small Diamonds

🔹 5. Streamlit Web Application

The deployed app allows users to:
Enter diamond properties
Predict diamond price using the trained regression model
Predict cluster segment using the trained KMeans model
View human-friendly cluster names
Get instant results through a clean, interactive UI
Models are loaded from .pkl files for fast and stable inference.

🛠️ Technologies Used

Python
Pandas, NumPy
Scikit-Learn
XGBoost
TensorFlow/Keras
Seaborn, Matplotlib
Streamlit
PCA
Joblib

📌 Conclusion

This project demonstrates a complete data science workflow—from raw data to a production-ready ML application.
It serves as a strong portfolio project for data science and ML engineering roles.
