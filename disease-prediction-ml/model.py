import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove extra spaces
data = data.apply(lambda col: col.astype(str).str.strip())
data = data.replace("nan", "None")

# Features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Encode features
for column in X.columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.savefig("confusion_matrix.png")
plt.show()