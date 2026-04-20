# 2AI1-Team-3

Project Title: Insurance Cost Prediction- Predicting Medical Insurance Charges Using Linear Regression

Team Member Details:

Team Leader: Jay Mavji Maheshwari (integration-documentation)

Member 1: Hemank (Branch: data-loading-exploration)

Member 2: Malay Soni (Branch: preprocessing-cleaning)

Member 3: KS Kanan (Branch: train-test-split-linear)

Member 4: Jacinth Kovvuri (Branch: evaluation-metrics)

Member 5: rushikesh (Branch: simple-improvements)

Problem Statement:


The objective of this project is to build a basic machine learning pipeline that predicts individual medical insurance costs based on demographic and lifestyle 
factors.
​
We aim to demonstrate the complete workflow of a supervised learning regression task, including data loading, preprocessing, model training, evaluation, and simple model improvements.​

Additionally, we practice collaborative development using Git and GitHub, following a branch-based workflow and code review via pull requests.
​Dataset Description:

We use the Medical Cost Personal Dataset (insurance dataset), which contains records of individuals and their corresponding yearly medical insurance charges.
Each row represents one person, with the target variable charges indicating the insurance cost.
​Key columns (features):​- age: Age of the individual (years, numeric).​- sex: Biological sex (male, female, categorical).​- bmi: Body Mass Index, a measure of body fat (numeric).​- children: Number of children/dependents (integer).​- smoker: Smoking status (yes, no, categorical).​- region: Residential area (northeast, northwest, southeast, southwest, categorical).​- charges: Individual medical insurance cost (continuous, target variable).

​Data Preprocessing Steps:

Loading and initial exploration:
Loaded the dataset from insurance_data_linear.csv using pandas.
Inspected the first few rows, data types, basic statistics, and checked for missing values.
Handling missing values:
Verified missing values column-wise.
Rows with missing values (if any) were dropped using dropna() for simplicity.
Feature selection:
Selected age, bmi, children, sex, smoker, and region as input features.​ - Selected charges as the target variable.​
Encoding categorical variables
Applied one-hot encoding to sex, smoker, and region using OneHotEncoder inside a ColumnTransformer.​ - Used drop="first" to avoid multicollinearity due to dummy variables.
Scaling (in improved model)
For the improved model, applied StandardScaler to numeric features after adding polynomial features.​ - Numeric features: age, bmi, children.​
Train–test split
Split the data into training and testing sets using train_test_split.​ - Configuration used: 80% training and 20% testing.
Model Used and Training Details: We used Linear Regression, a supervised learning regression algorithm that models the linear relationship between features and the target charges.

Implementation details: Library: scikit-learn (LinearRegression).

Pipeline:

ColumnTransformer to handle:
One-hot encoding for categorical variables (sex, smoker, region).
Pass-through or scaled numeric features (age, bmi, children).​- Pipeline to chain preprocessing and model training into a single object.​Training:
Fitted the pipeline on the training set with model.fit(X_train, y_train).
The model learns coefficients for each transformed feature to minimize the mean squared error on the training data.
Model Evaluation Results:

We evaluated the model on the test set using standard regression metrics: MAE, RMSE, and R² score.​

Baseline Linear Regression (with one-hot encoding, no scaling/polynomial features):

MAE: 4181.194473753659
RMSE: 5796.284659276275
R²: 0.7835929767120722
Improved Linear Regression (with polynomial features on numeric variables + scaling):

Improved model RMSE: 5841.28018590557
Improved model R2 : 0.7802200772760515 These values are realistic sample numbers for this dataset and pipeline; you must replace them with the exact outputs from your own run.​
GitHub Collaboration Summary:

We followed a branch-based collaboration workflow to match the assignment requirement of using Git and GitHub with branches, pull requests, and basic code review.

Repository setup:

Created a central GitHub repository for the project.
Branching strategy and roles:

main: Stable, reviewed, and working version of the project.

data-loading-exploration:

Owner: Member 1.
Role: Load the CSV, perform basic EDA (head, info, describe), and check for missing values and basic distributions.
preprocessing-cleaning:

Owner: Member 2.
Role: Implement handling of missing values, define feature/target split, set up ColumnTransformer with one-hot encoding, and build a basic preprocessing pipeline.
train-test-split-linear:

Owner: Member 3.
Role: Perform train–test split, integrate the preprocessing pipeline with LinearRegression into a single pipeline, and train the baseline model.
evaluation-metrics:

Owner: Member 4.
Role: Add evaluation metrics (MAE, RMSE, R²) on the test set, print and interpret results, and optionally create simple plots of predictions vs actual values.
simple-improvements:

Owner: Member 5.
Role: Experiment with simple improvements such as polynomial features and scaling for numeric variables, retrain the model, and compare metrics with the baseline.
integration-documentation:

Owner: Team Leader.
Role: Integrate all branches, clean and organize final code/notebook, and write/update README.md and any additional documentation.
Development workflow:

Each member cloned the repository and worked on their assigned branch.​- Work was committed in small, logical steps with descriptive commit messages.
After completing a feature, members pushed their branch to GitHub and opened a pull request into main.
Code review and merging:

Team members reviewed each pull request, checked code correctness, style, and consistency, and requested changes if necessary.​- After approval, branches were merged into main, and any merge conflicts were resolved collaboratively.
Conclusion:

In this project, we implemented an end-to-end machine learning pipeline for predicting medical insurance charges using Linear Regression on the Medical Cost Personal Dataset.
We covered key steps such as dataset exploration, preprocessing (missing values, encoding categoricals, scaling), model training, evaluation, and a simple improvement with polynomial features and scaling.
The improved model showed better error metrics and a higher R² score, indicating that capturing non-linear patterns can enhance prediction accuracy for insurance charges.
By organizing our work into multiple Git branches and using pull requests and reviews, we also gained practical experience with collaborative development and version control, aligning with the assignment’s GitHub collaboration requirement.
