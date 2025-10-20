import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load test data
test_data = pd.read_csv('data/test.csv')
X_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

# Load trained model
model = joblib.load('models/model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred))
}

# Save metrics to JSON
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Generate confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("Validation complete. Metrics saved to metrics.json")
