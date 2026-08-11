import pandas as pd
import numpy as np
import seaborn as sns  # Standard alias for Seaborn is 'sns', though 'sb' works
import sklearn # 'skilit' appears to be a typo for 'scikit-learn' or 'sklearn'
import matplotlib.pyplot as plt

#By Tanmay
print("This is the data")
# By Manas
print("Yes we know, these is data")
df = pd.read_csv('tested.csv')

print(df.head())

# step 1
# missing values 
print(f"missing value : {df['Age'].isnull().sum()}")

# in percent
print((df['Age'].isnull().sum() / len(df)) * 100)

# step 2
# replace the null with mean
# >calculate mean
mean = df['Age'].mean()
# >replace with mean
df['Age'] = df['Age'].fillna(mean)
# now check again 
print("Missing values in Age:", df['Age'].isnull().sum())
print(df['Age'])


# step 3
# scailing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# >making sure that the Fre not have nll values
# >if falre have null values
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# >selecting columnms to scale
columns_to_scale = ['Age', 'Fare']
# >operation
min_max_scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])

print("Normalized Data (0 to 1)")
print(df_normalized[columns_to_scale].head())

# setp 4
# encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_normalized['Sex'] = le.fit_transform(df_normalized['Sex'])

print(df_normalized['Sex'].head())

# step 5
# outiler detection and box plot
Q1 = df['Age'].quantile(0.25)  # 25th percentile
Q3 = df['Age'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1 

print(f"Q1 : {Q1}")
print(f"Q3 : {Q3}")

# Calculate IQR Boundaries
lower_boundary = Q1 - 1.5 * IQR
upper_boundary = Q3 + 1.5 * IQR

print(f"Lower limit for normal data: {lower_boundary}")
print(f"Upper limit for normal data: {upper_boundary}")

# lowest and highest points in the dataset
print(f"Youngest Passenger: {df['Age'].min()}")
print(f"Oldest Passenger: {df['Age'].max()}")

# box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_normalized, x='Age', color='skyblue')
plt.title('Box Plot of Age (Detecting Outliers)')
plt.show()
