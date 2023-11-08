import argparse
import re
import pickle
import contractions
import nltk
import pandas as pd

with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Define data preprocessing functions (clean_text and stemming)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'sent from.*', '', text)
    text = re.sub('\\n', ' ', text)
    text = re.sub('’', "'", text)

    text = contractions.fix(text)
    punctuations = '''()-[]{};:.'”“"\<>|./@#$%^&*_~'''
    for x in text:
        if x in punctuations:
            text = text.replace(x, "")
    text = re.sub(",", ' ', text)
    text = re.sub(r"\'s", ' ', text)
    text = re.sub(r'(.)\1\1+', r'\1', text)
    text = re.sub(r'\d+', "", text)
    text = text.strip()
    return text

def stemming(data):
    stemmer = nltk.PorterStemmer()
    tokens = nltk.word_tokenize(str(data))
    new_text = ''
    for t in tokens:
       new_text = new_text + ' '+ stemmer.stem(t)
    return new_text

# Load and preprocess input data from a CSV file
def load_input_data(input_file):
    data = pd.read_csv(input_file)

    # Apply preprocessing steps (clean_text)
    data['clean_text'] = data['text'].apply(lambda x: clean_text(x))
    data['stemmed'] = data['clean_text'].apply(lambda x: stemming(x))

    return  data['stemmed']

# Perform inference using the loaded model
def perform_inference(input_data):
    predictions = model.predict(input_data)
    return predictions

def save_predictions(predictions, output_file):
    data = pd.read_csv('test_reviews.csv')
    output = pd.DataFrame()
    output['id'] = data['id']
    output['sentiment'] = predictions
    output['sentiment'] = output['sentiment'].apply(lambda x : 'Positive' if x==1 else 'Negative')
    output.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    # Load and preprocess input data
    preprocessed_input_data = load_input_data(args.input)

    # Perform inference
    predictions = perform_inference(preprocessed_input_data)

    # Save predictions
    save_predictions(predictions, args.output)
