import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
train_data = pd.read_csv('data/train.csv')

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Train baseline model (LogisticRegression)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/model.pkl')

print("Model trained and saved to models/model.pkl")
