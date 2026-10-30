# Bank Marketing: Exploratory Data Analysis & Classification

**Author:** Yash Poddar (23BDS0195)  
**Course:** BCSE331L - Exploratory Data Analysis  
**Institution:** Vellore Institute of Technology (VIT)  

🎥 **[Project Video Demonstration](https://drive.google.com/file/d/1pifZ7ynp2JL1u4T1Gky-KPzQzcuJk-5N/view?usp=sharing)**

## Project Overview
This repository contains a comprehensive Exploratory Data Analysis (EDA) and Machine Learning pipeline applied to the **Bank Marketing Dataset**. The core objective of this project is to analyze client demographics, financial records, and campaign interactions to predict whether a client will subscribe to a term deposit. 

The project is structured into three distinct phases, scaling from basic data cleaning to advanced unsupervised clustering and supervised classification.

## Phase 1: Data Preprocessing & Visual Analysis (Modules 1-5)
* **Data Cleaning & Imputation:** Addressed missing data by dropping highly incomplete columns (e.g., `poutcome` with >74% missing values) and imputing categorical `unknown` strings using the mode.
* **Transformations:** Applied Log Transformations to correct extreme right-skewness in numerical features like `balance` and `duration`. Converted categorical target variables into binary flags for machine learning compatibility.
* **1D, 2D, and 3D Analysis:** 
  * **Univariate (1D):** Distribution plots of age, job types, and deposit subscriptions.
  * **Bivariate (2D):** Feature comparisons, highlighting that clients aged 60+ have the highest subscription rates.
  * **Multivariate (3D+):** Correlation heatmaps and pairplots to identify multicollinearity, revealing that call `duration` is the strongest predictor of a successful deposit.

## Phase 2: Unsupervised Learning & Clustering
* **Feature Scaling:** Applied `StandardScaler` to normalize continuous features (age, balance, duration, campaign, previous).
* **K-Means Clustering:** Utilized the **Elbow Method** to determine the optimal number of clusters ($k=3$) and partitioned the client base into distinct segments.
* **Dimensionality Reduction for Visualization:** Reduced the 5-dimensional scaled dataset to 2 dimensions using **PCA** to plot and visually validate the K-Means cluster separation.
* **Hierarchical Clustering:** Computed a linkage matrix (Ward's method) and generated a **Dendrogram** to confirm the hierarchical relationships before fitting an Agglomerative Clustering model.

## Phase 3: Dimensionality Reduction & Classification (Modules 6-7)
* **Principal Component Analysis (PCA):** Profiled the dataset using PCA, plotting the cumulative explained variance. Determined the exact number of principal components required to retain a 95% variance threshold.
* **Logistic Regression Classification:** 
  * Split the dataset into an 80-20 Train/Test split.
  * Trained a predictive Logistic Regression model to classify clients on the `deposit_flag`.
  * Evaluated model performance using Accuracy Scores, a Classification Report, and a plotted **Confusion Matrix** to break down true positive and true negative predictions.

## Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
* **Environment:** Google Colab / Jupyter Notebook
