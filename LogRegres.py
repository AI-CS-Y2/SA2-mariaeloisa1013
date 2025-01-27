import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# ------------------- DATASET -----------------------------

dataFrame = pd.read_csv('SDD.csv')

# ----------------- Data Cleaning ----------------------------

# LabelEncoder to convert into numeric data
Label = LabelEncoder()
dataFrame['Gender'] = Label.fit_transform(dataFrame['Gender'])
dataFrame['Study Satisfaction'] = Label.fit_transform(dataFrame['Study Satisfaction'])
dataFrame['Depression'] = Label.fit_transform(dataFrame['Depression'])
dataFrame['Dietary Habits'] = Label.fit_transform(dataFrame['Dietary Habits'])
dataFrame['Have you ever had suicidal thoughts ?'] = Label.fit_transform(dataFrame['Have you ever had suicidal thoughts ?'])

# categorizing sleep duration data into numerical values
def categorizeSleep(value):
    value = str(value).lower().strip()
    if "-" in value:
        return 1
    elif "less" in value:
        return 2
    elif "more" in value:
        return 0
    else:
        return np.nan

# add new categories into to dataframa
dataFrame["Cleaned Sleep Duration"] = dataFrame["Sleep Duration"].apply(categorizeSleep)

# ----------------- Choosing Attributes -------------

# assigning attributes(independent var) and predict(dependent var)
attributes = dataFrame[['Academic Pressure', 'Cleaned Sleep Duration', 'Study Satisfaction', 
               'Work/Study Hours', 'Dietary Habits', 'Financial Stress',
               'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']]
predict = dataFrame['Depression']

# ------------------ Check Missing Values -------------------

# imputation
numAttributes = attributes.select_dtypes(include=[np.number])
categoryAttributes = attributes.select_dtypes(exclude=[np.number])

# find missing values such as hidden NANs
imputer = SimpleImputer(strategy='mean')
numAttributes = pd.DataFrame(imputer.fit_transform(numAttributes), columns=numAttributes.columns)
categoryImputer = SimpleImputer(strategy='most_frequent')
categoryAttributes = pd.DataFrame(categoryImputer.fit_transform(categoryAttributes), columns=categoryAttributes.columns)

# finalize
attributes = pd.concat([numAttributes, categoryAttributes], axis=1)

# -------------- Splitting Data for Train/Test -------------------

# training data and testing data
X_train, X_test, y_train, y_test = train_test_split(attributes, predict, test_size=0.3, random_state=42)

# -------------------- Scaling Numbers -------------------

# scaling range of values
scaler = StandardScaler()
scaledXtrian = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
scaledXtest = scaler.transform(X_test.select_dtypes(include=[np.number]))

# -------------- LOGISTIC REGRESSION MODEL --------------------

logRegression = LogisticRegression(random_state=42, max_iter=1000)
logRegression.fit(scaledXtrian, y_train)

# ------------------ metrics ---------------------

# make the predictions
LR_Yprediction = logRegression.predict(scaledXtest)

# calculation
accuracy = accuracy_score(y_test, LR_Yprediction)
confusionMatrix = confusion_matrix(y_test, LR_Yprediction)
classificationReport = classification_report(y_test, LR_Yprediction)

ROCauc = roc_auc_score(y_test, logRegression.predict_proba(scaledXtest)[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, logRegression.predict_proba(scaledXtest)[:, 1])
ROCdata = pd.DataFrame({
    "Threshold": thresholds,
    "False Positive Rate (FPR)": fpr,
    "True Positive Rate (TPR)": tpr
})

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC-AUC: {ROCauc:.4f}')
print('Confusion Matrix:')
print(confusionMatrix)
print('Classification Report:')
print(classificationReport)
print(f"Area Under Curve (AUC): {ROCauc:.4f}")
print("ROC Data (Thresholds, FPR, TPR):")
print(ROCdata.head(10)) 

# ---------------- metric graphs -----------------

# classification report
report = classification_report(y_test, LR_Yprediction, output_dict=True)
metrics = pd.DataFrame(report).transpose()

metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', color=['red', 'green', 'blue'])
plt.title("CLASSIFICATION REPORT")
plt.xlabel("class")
plt.ylabel("score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='upper right')
plt.show()

# ROC 
plt.title('ROC CURVE')
plt.plot(fpr, tpr, label=f'Logistic Regression \n(AUC = {ROCauc:.4f})', color='red')
plt.plot([0, 1], [0, 1], 'g--', label='Random Guessing')
plt.xlabel('FP rate') #false positive
plt.ylabel('TP Rate') #true positive
plt.legend(loc='upper right')
plt.grid()
plt.show()

# ------------------- False Positives and False Negatives -------------------

# Create a DataFrame with predictions
predictionsDF = X_test.copy()
predictionsDF['Actual'] = y_test
predictionsDF['Predicted'] = LR_Yprediction

# False Positives (FP) and False Negatives (FN)
falsePositives = predictionsDF[(predictionsDF['Actual'] == 0) & (predictionsDF['Predicted'] == 1)]
falseNegatives = predictionsDF[(predictionsDF['Actual'] == 1) & (predictionsDF['Predicted'] == 0)]

# Display FP and FN
print("False Positives (FP) - First few rows:")
print(falsePositives.head())
print("\nFalse Negatives (FN) - First few rows:")
print(falseNegatives.head())
