# Analysis-on-Sales-Data-to-Calculate-and-Predict-Customer-Lifetime-Value-CLTV-with-Machine-Leaning
This repository contains a Jupyter Notebook for analyzing sales data and predicting Customer Lifetime Value (CLTV) using machine learning techniques. The project utilizes the 'Turkish Market Sales' dataset from Kaggle to estimate CLTV, compute RFM (Recency, Frequency, Monetary) metrics, and forecast customer behavior, ultimately providing actionable insights for business growth and retention strategies.

## Table of Contents
- [Overview](#overview)
- [Key Objectives](#key-objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Work Flow](#work-flow)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
Customer Lifetime Value (CLTV) is a vital metric for understanding the long-term value of a customer to a business. This project:

- Analyzes the **Turkish Market Sales** dataset to compute CLTV.
- Applies **machine learning models** to predict customer behaviour and improve business strategies.
- Includes steps like data cleaning, feature engineering, and model evaluation to ensure robust results.
The primary objective is to develop a predictive model that estimates CLTV and identifies opportunities for improving customer retention and profitability.

## Key Objectives

The main objectives of this project are:
1. **Calculate CLTV:** Estimate the Customer Lifetime Value (CLTV) for each customer to understand their long-term impact on the business.
2. **Compute RFM Metrics:** Compute the Recency, Frequency, and Monetary (RFM) metrics to evaluate customer activity and behavior over time.
3. **Predict Customer Behavior:** Use machine learning models to predict customer behavior, including future purchases and potential churn.
4. **Improve Retention Strategies:** Identify high-value customers and potential churn risks, enabling businesses to tailor retention strategies.
5. **Segmentation and Clustering:** Segment customers into distinct groups based on behavior and characteristics to further refine marketing strategies and personalization efforts.
6. **Evaluate Model Performance:** Assess the predictive accuracy of different machine learning models, such as KNN, Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
  
## Dataset
The Turkish Market Sales dataset from Kaggle is used in this project. It contains anonymized transaction data for customers in the Turkish market, which includes details about their purchases. Key features of the dataset include:

- **Customer ID:** Anonymized identifier for each customer.
- **Timestamp:** Date and time of the transaction.
- **Purchase Amount:** The total amount spent by the customer in a given transaction.
- **Product Categories:** Various product categories purchased by customers.
- **Customer Demographics:** (If available) Additional demographic information like age or region.

This dataset provides the necessary information to calculate important customer metrics such as RFM (Recency, Frequency, Monetary), which are key to estimating CLTV. To use this dataset:
- Download the [Turkish Market Sales dataset](https://www.kaggle.com/datasets/omercolakoglu/turkish-market-sales-dataset-with-9000items) from Kaggle (Please find the link in the `Data` folder).
- Place the dataset in the project directory, ensuring the dataset files are named correctly as referenced in the code.

> **Note**: This dataset contains anonymized transaction data, making it ideal for training machine learning models while ensuring customer privacy.

### Preprocessing Steps
- Removal of duplicate entries
- Filtering of sparse user-product interactions
- Normalization and preparation for models

## Technologies Used
The project is implemented using:
- **Programming Language:** Python
- **Libraries and Frameworks:**
   - scikit-learn
   - Pandas
   - Matplotlib
   - Seaborn
   - NumPy
   - XGBoost
- **Jupyter Notebook:** For analysis and visualization

## Work Flow
1. **Import Libraries and Data:** Load necessary libraries and the Kaggle dataset.
2. **Exploratory Data Analysis (EDA):** Visualize and understand data patterns.
3. **Data Cleaning:** Handle missing values, outliers, and inconsistencies.
4. **Lifetime Value Prediction:**
   - Compute RFM metrics for one month of data.
   - Forecast the next two months of customer behavior using RFM scores.
5. **Data Preprocessing and Feature Engineering:** Transform raw data into model-ready formats.
6. **Machine Learning Models:** Various models are used to predict customer behavior:
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
7. **Model Evaluation:** Models are evaluated using precision, recall, F1-score, and accuracy metrics.
8. **Cluster Analysis:** Customers are segmented to identify key behavior patterns.

## Results and Evaluation
### Performance of Models
| Model              | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| K-Nearest Neighbors | 0.65      | 0.75   | 0.69     | 75.0%    |
| Logistic Regression | 0.61      | 0.78   | 0.69     | 78.3%    |
| Random Forest       | 0.65      | 0.75   | 0.69     | 75.2%    |
| Gradient Boosting   | 0.63      | 0.78   | 0.69     | 78.2%    |
| XGBoost             | 0.66      | 0.78   | 0.69     | 78.0%    |

### Best Model
Among the models evaluated, Logistic Regression emerged as the best-performing model, attaining the highest accuracy of 78.3%. It also maintained a high recall and balanced F1-score. The high accuracy and strong recall indicate that Logistic Regression is highly effective at correctly identifying positive cases with reliability. Other models, such as Gradient Boosting and XGBoost, also performed admirably, achieving accuracies of 78.2% and 78.0%, respectively, making them strong alternatives. However, Logistic Regression stands out for its balance between precision, recall, and overall accuracy, making it the most suitable model for this classification task.
 
### Key Insights
- **Model Performance:** The **Logistic Regression** and **Gradient Boosting** models achieved the best accuracy, with scores of **78.3%** and **78.2%**, respectively, indicating they are the most effective for predicting CLTV in this dataset.
- **Customer Segmentation:** The largest customer segment (**Cluster 0**) represents **78.4%** of the total customer base. This indicates a high concentration of customer behavior within a few key segments, which businesses can target for tailored retention efforts.
- **Model vs Baseline Accuracy:** The model's performance closely aligns with the baseline accuracy of **78.4%**, suggesting the model performs well but can still be optimized for better predictive power.
- **Business Implications:** The insights gained from the customer behavior prediction and segmentation can be used to enhance marketing strategies, identify high-value customers, and mitigate churn risk by targeting potential at-risk customers.
- **Opportunities for Improvement:** Further optimization of the models can include hyperparameter tuning, feature engineering, and leveraging advanced ensemble techniques to enhance predictive accuracy.

## Usage
To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Analysis-on-Sales-Data-to-Calculate-and-Predict-Customer-Lifetime-Value-CLTV-with-Machine-Leaning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Analysis-on-Sales-Data-to-Calculate-and-Predict-Customer-Lifetime-Value-CLTV-with-Machine-Leaning
   ```
3. Install Dependencies:
Make sure you have Python installed. Then install the required libraries:
   ```bash
   pip install requirements.txt
   ```
4. Run the Notebook:
Open the Jupyter Notebook and execute the cells
   ```bash
   jupyter notebook Analysis_on_Sales_Data_to_Calculate_and_Predict_Customer_Lifetime_Value_(CLTV)_with_Machine_Leaning.ipynb
   ```
5. Ensure the dataset is available in the project directory.
6. Run the cells sequentially to execute the analysis.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, make changes, and submit a pull request. Please ensure your code adheres to the project structure and is well-documented.

