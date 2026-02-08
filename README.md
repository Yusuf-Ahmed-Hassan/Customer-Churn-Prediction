# Customer-Churn-Prediction
This project aims to predict customer churn for a bank — that is, whether a customer will leave or stay — using machine learning techniques. We compare two popular classification algorithms: Logistic Regression and K-Nearest Neighbors (KNN).

Problem Overview:
The goal is to classify customers into two categories:
Exited = 1: The customer left the bank
Exited = 0: The customer stayed with the bank

Dataset Features: 
RowNumber, CustomerId, Surname, CreditScore,
Geography, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary.

Target:
Exited (1 or 0)

Workflow & Implementation

1. Data Cleaning & Preprocessing
Dropped irrelevant features: RowNumber, CustomerId, and Surname
Handled categorical data using Label Encoding for Gender and Geography
Verified no missing values or duplicate records
Feature Scaling: Applied StandardScaler to normalize the data (essential for distance-based models like KNN)

2. Exploratory Data Analysis (EDA)
Generated a correlation heatmap to identify relationships between features and the target variable
Visualized the distribution of churners vs. non-churners

3. Machine Learning Models
Logistic Regression: Baseline linear model
K-Nearest Neighbors (KNN): Used with k=5 to capture non-linear patterns

Results & Evaluation
Performance was evaluated using Confusion Matrices and Classification Reports.
Model	Key Observation
Logistic Regression	Performed well in predicting staying customers but slightly struggled with churners
KNN	Generally higher accuracy; effectively identifies clusters of similar customer behaviors

Final Analysis:
Performance KNN: Has higher Accuracy. It is good at knowing who will STAY.
Logistic Regression: Is better at catching the customers who will LEAVE.
Why Logistic Regression is better for the Bank? In this project, we care about catching people before they leave.
The Problem with KNN: It is like a follower; it just looks at what the "neighbors" do. If most people stay, it will always guess "Stay," so it misses the people who leave.
The Advantage of Logistic: It gives us a Reason. It tells the bank: "Customers leave because they are Older or have High Balances." This helps the bank fix the problem.
Conclusion The Logistic Regression is the winner for this project.
Why? Because for a bank, missing a customer who is about to leave is a big loss. Logistic Regression is more "sensitive" to these customers than KNN.

How to Run:
Ensure you have the following libraries installed:
pip install pandas numpy matplotlib seaborn scikit-learn
Place the Churn Modeling.csv file in the same directory as the script.
Run the Jupyter Notebook or Python script.
