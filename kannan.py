# branch: train-test-split-linear-model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

df = pd.read_csv("insurance_data_linear.csv")
df = df.dropna()

X = df.drop("charges", axis=1)
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()

model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("model", lin_reg)])

model.fit(X_train, y_train)

print("Linear Regression model trained.")
print("Train size:", X_train.shape, "Test size:", X_test.shape)