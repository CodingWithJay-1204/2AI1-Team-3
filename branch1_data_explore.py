# branch: data-loading-exploration

import pandas as pd

# 1. Load dataset
df = pd.read_csv("insurance_data_linear.csv")

# 2. Basic info
print(df.head())
print(df.info())
print(df.describe())

# 3. Check missing values
print(df.isnull().sum())

# 4. Simple EDA
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df["region"].value_counts())

import matplotlib.pyplot as plt

plt.hist(df["charges"], bins=30)
plt.xlabel("Charges")
plt.ylabel("Count")
plt.title("Distribution of insurance charges")
plt.show()