#THIS IS STRAIGHT FROM DEEPSEEK I WAS TESTING TO SEE IF IT WORKED
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')

# Load the dataset
file = "labeled_data.csv"
data = pd.read_csv(file)

# Print column names to verify
print("Column names:", data.columns)

# Clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Replace 'tweet' with the correct column name
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['class']  # Replace 'class' with the correct column name

# Save the preprocessed data and vectorizer
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump((X, y), f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Data preparation complete. Preprocessed data saved as 'preprocessor.pkl' and vectorizer saved as 'vectorizer.pkl'")