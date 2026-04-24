import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true])

# 🔥 Shuffle VERY IMPORTANT
data = data.sample(frac=1, random_state=42)

# 🔥 Combine title + text
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 🔥 Accuracy check
X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)
print("Accuracy:", accuracy)

# Save
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ MODEL READY")