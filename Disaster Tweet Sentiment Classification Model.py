# Step 1: Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load the dataset
df = pd.read_csv("disaster_tweets_data(DS).csv")

# Step 3: Handle null values
df.dropna(inplace=True)

# Step 4: Preprocess the disaster tweets data
def preprocess_text(text):
    # a) Remove URLs and punctuations, b) Convert to lowercase, c) Remove extra spaces
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", '', text)  # Remove URLs & punctuations
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to the 'tweets' column
df['cleaned_tweets'] = df['tweets'].apply(preprocess_text)

# Step 5: Transform the words into vectors using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_tweets'])

# Step 6: Select X (independent feature) and y (dependent feature)
y = df['target']

# Step 7: Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Apply models and generate predictions
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN Classification': KNeighborsClassifier(n_neighbors=5)
}

# Step 9: Predict, compute confusion matrix, classification report, and best accuracy
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)
    print("Accuracy:", acc)

    # Track the best accuracy model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model_name

print(f"\n✅ Best model based on accuracy: {best_model} with accuracy = {best_accuracy:.4f}")
