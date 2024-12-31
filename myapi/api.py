from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import nltk
import string
import contractions
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize NLTK tools
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wn = WordNetLemmatizer()

# Define the text preprocessing function
def text_preprocessing(mystr):
    mystr = mystr.lower()  # Case folding
    mystr = re.sub(r'\w*\d\w*', '', mystr)  # Remove words with digits
    mystr = re.sub(r'\n', ' ', mystr)  # Replace new line characters with space
    mystr = re.sub(r'[‘’“”…]', '', mystr)  # Remove quotes
    mystr = re.sub(r'<.*?>', '', mystr)  # Remove HTML tags
    mystr = re.sub(r'\[.*?\]', '', mystr)  # Remove text in square brackets
    mystr = re.sub(r'https?://\S+|www\.\S+', '', mystr)  # Remove URLs
    mystr = clean(mystr, no_emoji=True)  # Remove emojis
    mystr = ''.join([c for c in mystr if c not in string.punctuation])  # Remove punctuation
    mystr = ' '.join([contractions.fix(word) for word in mystr.split()])  # Expand contractions
    
    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(mystr)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [wn.lemmatize(token) for token in tokens]
    new_str = ' '.join(tokens)
    return new_str.lower()

# Create Flask App
app = Flask(__name__)

# Load the saved model (Pickle file)
model = joblib.load("../r_model.pkl")

# Create API routing call
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request
    new_data = request.json

    # Convert JSON request to Pandas DataFrame
    df = pd.DataFrame(new_data)

    # Apply text preprocessing on each review in the DataFrame
    df['review'] = df['review'].apply(text_preprocessing)

    # Get prediction
    prediction = model.predict(df)

    # Return JSON version of the prediction
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
