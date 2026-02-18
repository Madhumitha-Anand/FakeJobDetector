import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Load dataset
# -----------------------------------
df = pd.read_csv("data.csv")

# Drop missing values just in case
df = df.dropna(subset=["description", "fraudulent"])

X = df["description"]
y = df["fraudulent"]

# -----------------------------------
# Train-test split (STRATIFIED)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# ML Pipeline
# -----------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

# -----------------------------------
# Train model
# -----------------------------------
pipeline.fit(X_train, y_train)

# -----------------------------------
# Evaluate model
# -----------------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# Save model
# -----------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump((pipeline, accuracy), f)

print("\n✅ Model trained and saved as model.pkl")
