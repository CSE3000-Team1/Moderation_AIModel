import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')

file = "labeled_data.csv"
data = pd.read_csv(file)

print("Column names: ", data.columns)

def clean_text(text):
    text = re.sub(r'\W', ' ', text)   #remove special characters
    text = text.lower()                #convert text to lowercase
    text = re.sub(r'\s+', ' ', text)   #remove extra spaces
    return text

data['cleaned_text'] = data['tweet'].apply(clean_text)   #replace 'text' with column name

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

#convert text to numerical features
vectorizer = TfidfVectorizer(max_features = 5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['class']  #replace 'label' with column name

#save the vectorizer
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump((X,y), f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Data preparation complete. Preprocessed data saved as 'preprocessor.pkl' and vectorizer saved as 'vectorizer.pkl'")