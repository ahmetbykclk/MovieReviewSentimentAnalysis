import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load the movie reviews dataset (assuming it's in CSV format with columns "Review" and "Sentiment")
data = pd.read_csv('dataset.csv')

# Preprocess the text data (optional, you can add more preprocessing steps if needed)
data['review'] = data['review'].apply(lambda x: x.lower())

# Initialize the TF-IDF vectorizer and fit it to the review text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)
print(classification_report(y_val, val_predictions))

# Function to predict the sentiment of a new review
def predict_sentiment(new_review):
    new_review = new_review.lower()
    new_review_vectorized = tfidf_vectorizer.transform([new_review])
    predicted_sentiment = model.predict(new_review_vectorized)[0]
    return predicted_sentiment

# Print the first 10 test values, their predicted sentiment, and actual sentiment
print("\nFirst 10 Test Values:")
for i in range(10):
    review = data['review'][i]
    actual_sentiment = data['sentiment'][i]
    predicted_sentiment = predict_sentiment(review)
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predicted_sentiment}")
    print(f"Actual Sentiment: {actual_sentiment}")
    print("------")

# Test the model with a new review
new_review = "This movie is amazing! I highly recommend it."
predicted_sentiment = predict_sentiment(new_review)
print(f"\nReview: {new_review}")
print(f"Predicted Sentiment: {predicted_sentiment}")
