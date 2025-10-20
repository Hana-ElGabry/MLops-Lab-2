import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training data
train_data = pd.read_csv('data/train.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Train baseline model (LogisticRegression)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/model.pkl')

print("Model trained and saved to models/model.pkl")
