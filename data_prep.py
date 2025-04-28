# This script preprocesses the data for a sentiment analysis model.
# It cleans the text data, removes stopwords, and converts the text to numerical features using TF-IDF vectorization.
# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# load the dataset
file = "labeled_data.csv"
data = pd.read_csv(file)

# check the first few rows of the dataset
print("Column names: ", data.columns)

#function to clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)   #remove special characters
    text = text.lower()                #convert text to lowercase
    text = re.sub(r'\s+', ' ', text)   #remove extra spaces
    return text

data['cleaned_text'] = data['tweet'].apply(clean_text)   #replace 'text' with column name

#remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

#apply remove_stopwords function to cleaned_text column
data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

#convert text to numerical features
vectorizer = TfidfVectorizer(max_features = 5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['class']  #replace 'label' with column name

#save the vectorizer
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump((X,y), f)

#save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

#give confirmation message
print("Data preparation complete. Preprocessed data saved as 'preprocessor.pkl' and vectorizer saved as 'vectorizer.pkl'")