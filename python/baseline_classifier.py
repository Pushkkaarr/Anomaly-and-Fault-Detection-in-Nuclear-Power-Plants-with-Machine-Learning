import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
print("Loading Data...")
df = pd.read_csv('nuclear_fault_data_v2.csv')

# 2. Features vs Labels
# We use T, P, F, R to predict the Label
X = df[['Power', 'Fuel_Temp', 'Coolant_Temp', 'Pressure', 'Flow']]
y = df['Label']

# 3. Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate
print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Scram', 'LOFA']))

# 6. Confusion Matrix (The Proof)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Scram', 'LOFA'],
            yticklabels=['Normal', 'Scram', 'LOFA'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Baseline Fault Diagnosis')
plt.savefig('baseline_results.png')
print("Saved result graph to 'baseline_results.png'")