import streamlit as st
import joblib
import re
import nltk
import string
import contractions
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wn = WordNetLemmatizer()

# Define or import the 'text_preprocessing' function here
def text_preprocessing(mystr):
    mystr = mystr.lower()  # Case folding
    mystr = re.sub('\w*\d\w*', '', mystr)  # Remove digits
    mystr = re.sub('\n', ' ', mystr)  # Replace new line characters with space
    mystr = re.sub('[‘’“”…]', '', mystr)  # Remove quotes
    mystr = re.sub('<.*?>', '', mystr)  # Remove HTML tags
    mystr = re.sub(r'\[.*?\]', '', mystr)  # Remove text in square brackets
    mystr = re.sub('https?://\S+|www.\.\S+', '', mystr)  # Remove URLs
    mystr = clean(mystr, no_emoji=True)  # Remove emojis
    mystr = ''.join([c for c in mystr if c not in string.punctuation])  # Remove punctuation
    mystr = ' '.join([contractions.fix(word) for word in mystr.split()])  # Expand contractions
    
    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(mystr)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [wn.lemmatize(token) for token in tokens]
    new_str =  ' '.join(tokens)
    return new_str.lower()
    

# Streamlit App title
st.title("Sentiment Analysis Web App")

# Input text for the movie review
input_text = st.text_area("Enter a movie review:")

# Try loading the model with the custom 'text_preprocessing' function
try:
    model = joblib.load('../r_model.pkl')
    # Comment out or remove this line if you don't want to display the success message
    # st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading the model: {e}")
    model = None

# Predict button and displaying results, only if the model is loaded
if model:
    if st.button("Predict Sentiment"):
        if input_text:
            try:
                prediction = model.predict([input_text])[0]
                st.write(f"The predicted sentiment is: {prediction}")
            except Exception as e:
                st.write(f"Error making prediction: {e}")
        else:
            st.write("Please enter a movie review to analyze sentiment.")
else:
    st.write("Model not loaded, unable to make predictions.")