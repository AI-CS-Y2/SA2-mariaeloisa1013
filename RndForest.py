import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ------------------- DATASET -----------------------------

dataFrame = pd.read_csv('SDD.csv')  

# ------------------- Data Cleaning -------------------

# manually categorize category values into numeric
dataFrame['Family History of Mental Illness'] = dataFrame['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
dataFrame['Have you ever had suicidal thoughts ?'] = dataFrame['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
dataFrame['Dietary Habits'] = dataFrame['Dietary Habits'].map({'Unhealthy': 2, 'Moderate': 1, 'Healthy': 0})

# categorizing sleep duration data into numerical values
def ccategorizeSleep(value):
    value = str(value).lower().strip() 
    if "-" in value:
        return 1 
    elif "less" in value:
        return 2  
    elif "more" in value:
        return 0  
    else:
        return np.nan  # If the value doesn't match, return NaN

# add new categories into to dataframa
dataFrame["Cleaned Sleep Duration"] = dataFrame["Sleep Duration"].apply(ccategorizeSleep)

# ----------------- Choosing Attributes -------------


attributes = dataFrame[['Academic Pressure', 'Cleaned Sleep Duration', 'Study Satisfaction', 
               'Work/Study Hours', 'Dietary Habits', 'Financial Stress',
               'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']]
predict = dataFrame['Depression']

# -------------- Splitting Data for Train/Test -------------------

X_train, X_test, y_train, y_test = train_test_split(attributes, predict, test_size=0.3, random_state=42)

# ------------------- RANDOM FOREST MODEL -------------------

randomForest = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
randomForest.fit(X_train, y_train)  

# ------------------- Predictions -------------------
RF_Yprediction = randomForest.predict(X_test)

# --------------------- metrics -------------------

# predictions
Yprobability = randomForest.predict_proba(X_test)[:, 1]  

# calculations
accuracy = accuracy_score(y_test, RF_Yprediction)
confusionMetrix = confusion_matrix(y_test, RF_Yprediction)
classificationReport = classification_report(y_test, RF_Yprediction)

fpr, tpr, thresholds = roc_curve(y_test, Yprobability) # calculate ROC 
ROCauc = auc(fpr, tpr) # calculate AUC 
ROCdata = pd.DataFrame({
    'Threshold': thresholds,
    'False Positive Rate (FPR)': fpr,
    'True Positive Rate (TPR)': tpr
})

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(confusionMetrix)
print('Classification Report:')
print(classificationReport)
print(f"Area Under Curve (AUC): {ROCauc:.4f}")
print("\nROC Data (Thresholds, FPR, TPR):")
print(ROCdata.head(10))  

# ---------------- metric graphs -----------------

# classification report
report = classification_report(y_test, RF_Yprediction, output_dict=True)
metrics = pd.DataFrame(report).transpose()

metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', alpha=0.7,  color=['#00FFFF', 'magenta', 'yellow'])
plt.title('Classification Report - Random Forest')
plt.ylabel('Scores')
plt.xlabel('Classes')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ROC 
plt.plot(fpr, tpr, color='#00FFFF', lw=2, label=f'ROC curve (AUC = {ROCauc:.2f})')
plt.plot([0, 1], [0, 1], color='magenta', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

