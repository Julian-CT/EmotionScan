import pandas as pd
import re

EMOTIONS = ["AUSENTE", "RAIVA", "TRISTEZA", "MEDO", "CONFIANÇA", "ALEGRIA", "AMOR"]

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Normalize labels to binary vector per emotion
    def encode_labels(emotion_list):
        binary = [1 if emo in str(emotion_list) else 0 for emo in EMOTIONS]
        return binary
    
    df["Emotion_Label"] = df["Emotion_Label"].str.upper()
    df['encoded_emotions'] = df['Emotion_Label'].apply(encode_labels)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # links
        text = re.sub(r"@\w+|#", '', text)  # menções e hashtags
        text = re.sub(r"[^\w\s]", '', text)  # pontuação
        text = re.sub(r"\d+", '', text)  # números
        return text.strip()
    
    df['Text'] = df['Text'].apply(lambda x: preprocess(str(x)))

    return df['Text'].tolist(), df['encoded_emotions'].tolist()