import pandas as pd
import re

SENTIMENTS = ["NEGATIVO", "NEUTRO", "POSITIVO"]

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Normalize labels to one-hot vector
    def encode_sentiment(sentiment):
        sentiment = str(sentiment).upper()
        return [1 if sent == sentiment else 0 for sent in SENTIMENTS]
    
    df["Sentimento_Label"] = df["Sentimento_Label"].str.upper()
    df['encoded_sentiment'] = df['Sentimento_Label'].apply(encode_sentiment)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # links
        text = re.sub(r"@\w+|#", '', text)  # menções e hashtags
        text = re.sub(r"[^\w\s]", '', text)  # pontuação
        text = re.sub(r"\d+", '', text)  # números
        return text.strip()
    
    df['Text'] = df['Text'].apply(lambda x: preprocess(str(x)))

    return df['Text'].tolist(), df['encoded_sentiment'].tolist()