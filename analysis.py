import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------- DATASET -----------------------------
dataFrame = pd.read_csv('SDD.csv')

# initialize
columns_to_drop = ['Age', 'Job Satisfaction', 'City', 'Profession', 'Work Pressure', 'Degree']
dataFrame = dataFrame.drop(columns=columns_to_drop)

# --------------- analysis -------------------
# first 5 rows
print("\n\nfirst 5 rows:\n")
print(dataFrame.head())

# check missing values
print("\n\nmissing values:\n")
print(dataFrame.isnull().sum())

# basic statistics 
print("\n\n statistics:\n")
print(dataFrame.describe())

# data types
print("\n\n data types:\n")
print(dataFrame.dtypes)

# --------------- visualized statistics -------------------

# numerical charts
numericalAtts = dataFrame.select_dtypes(include=['float64', 'int64']).columns
dataFrame[numericalAtts].hist(bins=15, figsize=(10, 8))
plt.tight_layout()
plt.show()

# category charts
categoryAtts = dataFrame.select_dtypes(include=['object']).columns
for col in categoryAtts:
    value_counts = dataFrame[col].value_counts()
    value_counts.plot(kind='bar', title=f'{col} Bar Graph', color='lightblue')
    plt.xticks(rotation=45)
    plt.show()


# correlation matrix
correlationMatrix = dataFrame[numericalAtts].corr()
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(correlationMatrix, cmap='coolwarm')
fig.colorbar(cax)

for (i, j), val in np.ndenumerate(correlationMatrix):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')


plt.title("Correlation Matrix")
plt.xticks(range(len(numericalAtts)), numericalAtts, rotation=45)
plt.yticks(range(len(numericalAtts)), numericalAtts)
plt.show()
