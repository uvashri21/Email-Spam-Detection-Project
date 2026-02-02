import pandas as pd
import logging
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


DATA_PATH = Path("../data/spam.csv")
MODEL_PATH = Path("spam_model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")


try:
    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    logging.info("Dataset loaded successfully")
except FileNotFoundError:
    logging.error("Dataset not found. Check file path.")
    raise

# Handle common dataset format
if {'v1', 'v2'}.issubset(data.columns):
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
elif {'label', 'message'}.issubset(data.columns):
    pass
else:
    raise ValueError("Dataset must contain label & message columns")

# Map labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Remove missing values
data.dropna(inplace=True)

X = data['message']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.9
    )),
    ("classifier", MultinomialNB())
])


param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "classifier__alpha": [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

logging.info("Training model...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(best_model, model_file)

logging.info("Model saved successfully")

print("\n Advanced Spam Detection Model Training Completed")

