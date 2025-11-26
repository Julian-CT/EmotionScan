from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import json
import pickle
import re
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Add the API directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from your existing trained model
import torch
import numpy as np
from transformers import BertTokenizer

app = Flask(__name__)
CORS(app)

print("Starting Emotion and Sentiment Classification API...")

# Use absolute path to avoid issues with relative paths
# Handle both local development and Vercel deployment
if os.path.exists('/var/task'):
    # Vercel serverless environment
    BASE_DIR = '/var/task'
else:
    # Local development
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
BERT_EMOTIONS_PATH = os.path.join(MODELS_DIR, "bertimbau-mlp-ai")
BERT_SENTIMENT_PATH = os.path.join(MODELS_DIR, "bertimbau-mlp-sentiment")
MNB_MODEL_DIR = os.path.join(MODELS_DIR, "mnb")
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Labels
EMOTION_LABELS = ["AUSENTE", "RAIVA", "TRISTEZA", "MEDO", "CONFIANÇA", "ALEGRIA", "AMOR"]
SENTIMENT_LABELS = ["NEGATIVO", "NEUTRO", "POSITIVO"]

# Preprocessing
url_pat = re.compile(r"https?://\S+|www\.\S+")
mention_pat = re.compile(r"@\w+")

def preprocess_light(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = url_pat.sub("", text)
    text = mention_pat.sub("", text)
    return text.strip()

# ========== Load BERTimbau Emoções Model ==========
print(f"\n{'='*60}")
print("Loading BERTimbau Emoções Model...")
print(f"{'='*60}")

bert_emotions_model = None
bert_emotions_tokenizer = None

try:
    # Import BERTimbauMLP for emotions - ensure we use the correct path
    src_path = os.path.join(BERT_EMOTIONS_PATH, "src")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source path not found: {src_path}")
    
    # Use importlib to load from specific path
    import importlib.util
    model_file_path = os.path.join(src_path, "models", "bertimbau_mlp.py")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    
    spec = importlib.util.spec_from_file_location("bertimbau_mlp_emotions", model_file_path)
    bertimbau_emotions_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bertimbau_emotions_module)
    BERTimbauEmotionsMLP = bertimbau_emotions_module.BERTimbauMLP
    
    print(f"Loaded BERTimbauMLP from: {model_file_path}")
    
    # Load tokenizer
    bert_emotions_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    # Create model (7 classes for emotions)
    bert_emotions_model = BERTimbauEmotionsMLP().to(DEVICE)
    
    # Load weights - use bertimbau_mlp.pt for emotions
    model_path = os.path.join(BERT_EMOTIONS_PATH, "bertimbau_mlp.pt")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        bert_emotions_model.load_state_dict(state_dict)
        bert_emotions_model.eval()
        print("✅ BERTimbau Emoções model loaded successfully!")
        print(f"   Using: bertimbau_mlp.pt")
        
        # Test prediction to verify model is working
        test_text = "Estou com muita raiva dessa situação!"
        test_inputs = bert_emotions_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        test_inputs = {k: v.to(DEVICE) for k, v in test_inputs.items()}
        with torch.no_grad():
            test_outputs = bert_emotions_model(input_ids=test_inputs['input_ids'], attention_mask=test_inputs['attention_mask'])
            test_probs = test_outputs.cpu().numpy()[0]
        test_max_idx = np.argmax(test_probs)
        print(f"   Test prediction: '{test_text}' -> {EMOTION_LABELS[test_max_idx]} (prob: {test_probs[test_max_idx]:.4f})")
        print(f"   All test probs: {dict(zip(EMOTION_LABELS, test_probs))}")
    else:
        print(f"❌ Weights not found at: {model_path}")
        print(f"   Expected: {model_path}")
        bert_emotions_model = None
except Exception as e:
    print(f"❌ Error loading BERTimbau Emoções: {e}")
    import traceback
    traceback.print_exc()
    bert_emotions_model = None

# ========== Load BERTimbau Sentimentos Model ==========
print(f"\n{'='*60}")
print("Loading BERTimbau Sentimentos Model...")
print(f"{'='*60}")

bert_sentiment_model = None
bert_sentiment_tokenizer = None

try:
    # Import BERTimbauMLP for sentiment - ensure we use the correct path
    src_path = os.path.join(BERT_SENTIMENT_PATH, "src")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source path not found: {src_path}")
    
    # Use importlib to load from specific path
    import importlib.util
    model_file_path = os.path.join(src_path, "models", "bertimbau_mlp.py")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    
    spec = importlib.util.spec_from_file_location("bertimbau_mlp_sentiment", model_file_path)
    bertimbau_sentiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bertimbau_sentiment_module)
    BERTimbauSentimentMLP = bertimbau_sentiment_module.BERTimbauMLP
    
    print(f"Loaded BERTimbauMLP from: {model_file_path}")
    
    # Load tokenizer
    bert_sentiment_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    # Create model (3 classes for sentiment)
    bert_sentiment_model = BERTimbauSentimentMLP().to(DEVICE)
    
    # Load weights - use bertimbau_sentiment_best.pt for sentiment
    model_path = os.path.join(BERT_SENTIMENT_PATH, "bertimbau_sentiment_best.pt")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        bert_sentiment_model.load_state_dict(state_dict)
        bert_sentiment_model.eval()
        print("✅ BERTimbau Sentimentos model loaded successfully!")
        print(f"   Using: bertimbau_sentiment_best.pt")
    else:
        print(f"❌ Weights not found at: {model_path}")
        print(f"   Expected: {model_path}")
        bert_sentiment_model = None
except Exception as e:
    print(f"❌ Error loading BERTimbau Sentimentos: {e}")
    import traceback
    traceback.print_exc()
    bert_sentiment_model = None

# ========== Load MNB Sentimentos Model ==========
print(f"\n{'='*60}")
print("Loading MNB Sentimentos Model...")
print(f"{'='*60}")

mnb_model = None
mnb_vectorizer = None
mnb_mlb = None

try:
    # Load MNB components from models directory
    mnb_model_path = os.path.join(MNB_MODEL_DIR, "mnb_model.pkl")
    mnb_vectorizer_path = os.path.join(MNB_MODEL_DIR, "mnb_vectorizer.pkl")
    mnb_mlb_path = os.path.join(MNB_MODEL_DIR, "mnb_mlb.pkl")
    
    if os.path.exists(mnb_model_path) and os.path.exists(mnb_vectorizer_path) and os.path.exists(mnb_mlb_path):
        with open(mnb_model_path, 'rb') as f:
            mnb_model = pickle.load(f)
        with open(mnb_vectorizer_path, 'rb') as f:
            mnb_vectorizer = pickle.load(f)
        with open(mnb_mlb_path, 'rb') as f:
            mnb_mlb = pickle.load(f)
        print("✅ MNB Sentimentos model loaded successfully!")
        print(f"   Using: mnb_model.pkl, mnb_vectorizer.pkl, mnb_mlb.pkl")
    else:
        print(f"❌ MNB model files not found")
        print(f"   Expected:")
        print(f"     - {mnb_model_path}")
        print(f"     - {mnb_vectorizer_path}")
        print(f"     - {mnb_mlb_path}")
        mnb_model = None
        mnb_vectorizer = None
        mnb_mlb = None
except Exception as e:
    print(f"❌ Error loading MNB Sentimentos: {e}")
    import traceback
    traceback.print_exc()
    mnb_model = None
    mnb_vectorizer = None
    mnb_mlb = None

# ========== Preprocessing Functions ==========
def preprocess_text(text):
    """Preprocessing function for MNB Sentimentos"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # links
    text = re.sub(r"@\w+|#", '', text)  # mentions and hashtags
    text = re.sub(r"[^\w\s]", '', text)  # punctuation
    text = re.sub(r"\d+", '', text)  # numbers
    return text.strip()

# ========== Prediction Functions ==========
def predict_bert_emotions(text):
    """Predict using BERTimbau Emoções model"""
    if not bert_emotions_model:
        return None, None, None
 
    # Use text directly without preprocess_light, like the old version
    inputs = bert_emotions_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
 
    with torch.no_grad():
        outputs = bert_emotions_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Model already has sigmoid in the classifier, so outputs are already probabilities
        probs = outputs.cpu().numpy()[0]
 
    max_idx = np.argmax(probs)
    emotion = EMOTION_LABELS[max_idx]
    max_prob = probs[max_idx]

    # BERTimbau Emoções only returns emotions, not feelings
    # Return None for feeling to indicate it's not applicable
    return emotion, None, float(max_prob)

def predict_bert_sentiment(text):
    """Predict using BERTimbau Sentimentos model"""
    if not bert_sentiment_model:
        return None, None, None
    
    # Use text directly without preprocess_light
    inputs = bert_sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_sentiment_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # Use softmax for single-label
    
    max_idx = np.argmax(probs)
    sentiment_label = SENTIMENT_LABELS[max_idx]
    confidence = float(probs[max_idx])
    
    # Map to frontend format (capitalize first letter)
    sentiment_map = {
        "NEGATIVO": "Negativo",
        "NEUTRO": "Neutro",
        "POSITIVO": "Positivo"
    }
    feeling = sentiment_map.get(sentiment_label, sentiment_label.capitalize())
    
    # For sentiment models, emotion is AUSENTE and feeling is the sentiment
    return "AUSENTE", feeling, confidence

def predict_mnb(text):
    """Predict using MNB Sentimentos model"""
    if not mnb_model:
        return None, None, None
    
    processed_text = preprocess_text(text)
    X_vectorized = mnb_vectorizer.transform([processed_text])
    
    # Get probabilities
    probs = mnb_model.predict_proba(X_vectorized)[0]
    
    # Get predicted labels
    predictions = mnb_model.predict(X_vectorized)
    predicted_labels = mnb_mlb.inverse_transform(predictions)[0]
    
    if len(predicted_labels) > 0:
        sentiment_label = predicted_labels[0] if len(predicted_labels) == 1 else ';'.join(predicted_labels)
        max_prob = max(probs) if len(probs) > 0 else 0.0
    else:
        sentiment_label = "NEUTRO"
        max_prob = 0.0
    
    # Map to frontend format (capitalize first letter)
    sentiment_map = {
        "NEGATIVO": "Negativo",
        "NEUTRO": "Neutro",
        "POSITIVO": "Positivo"
    }
    feeling = sentiment_map.get(sentiment_label.upper(), sentiment_label.capitalize())
    
    # For MNB, emotion is AUSENTE and feeling is the sentiment
    return "AUSENTE", feeling, float(max_prob)

# ========== FRONTEND ROUTES ==========
# Serve frontend static files
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.route('/')
def serve_index():
    """Serve index.html"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend static files (CSS, JS, images, etc.)"""
    # Don't serve API routes as files
    if path.startswith('api/'):
        return None
    
    # Try to serve static file
    try:
        return send_from_directory(FRONTEND_DIR, path)
    except:
        # If file not found, serve index.html (for client-side routing)
        return send_from_directory(FRONTEND_DIR, 'index.html')

# ========== API ENDPOINTS ==========
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models": {
            "bert_emotions": bert_emotions_model is not None,
            "bert_sentiment": bert_sentiment_model is not None,
            "mnb": mnb_model is not None
        }
    })

@app.route('/api/predict_csv', methods=['POST'])
def predict_csv():
    try:
        print("Received CSV prediction request...")
 
        # Get model type from query parameter
        model_type = request.args.get('model', 'bert-emotions').lower()
        print(f"Using model: {model_type}")
 
        texts = []

        file_storage = request.files.get('file')
        if file_storage and file_storage.filename:
            filename = file_storage.filename
            filename_lower = filename.lower()
            print(f"Processing uploaded file: {filename_lower}")
            try:
                if filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
                    file_storage.stream.seek(0)
                    df = pd.read_excel(file_storage)
                elif filename_lower.endswith('.csv'):
                    file_storage.stream.seek(0)
                    file_bytes = file_storage.read()
                    df = None
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    separators = [',', ';', '\t']

                    for enc in encodings:
                        try:
                            text = file_bytes.decode(enc)
                        except UnicodeDecodeError:
                            continue

                        for sep in separators:
                            buffer = io.StringIO(text)
                            kwargs = {}
                            if sep != ',':
                                kwargs['sep'] = sep
                                kwargs['engine'] = 'python'

                            try:
                                df = pd.read_csv(buffer, **kwargs)
                                if df is not None:
                                    break
                            except Exception:
                                continue

                        if df is not None:
                            break

                    if df is None:
                        raise ValueError("Não foi possível decodificar o arquivo CSV com os encodings testados (utf-8, latin-1, iso-8859-1, cp1252)")
                else:
                    return jsonify({"error": "Formato de arquivo não suportado. Use .csv, .xls ou .xlsx"}), 400
            except Exception as e:
                print(f"Error reading file: {e}")
                return jsonify({"error": f"Erro ao ler arquivo: {str(e)}"}), 400

            text_column = None
            for candidate in ['text', 'Text', 'texto', 'Texto', 'content', 'Content']:
                if candidate in df.columns:
                    text_column = candidate
                    break

            if not text_column:
                print(f"Columns available: {df.columns.tolist()}")
                return jsonify({"error": "Arquivo deve conter uma coluna chamada 'text'"}), 400

            texts = df[text_column].dropna().astype(str).tolist()
        else:
            # Fallback: raw CSV content (compatibilidade com versões anteriores)
            csv_data = request.get_data(as_text=True)
            print(f"Raw CSV data length: {len(csv_data)} characters")

            lines = csv_data.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    if i == 0 and line.strip().lower() in ['text', 'texto', 'content']:
                        continue

                    parts = line.split(';')
                    if len(parts) >= 1:
                        text = parts[0].strip('"')
                        if text.lower() not in ['text', 'texto', 'content', '']:
                            texts.append(text)
 
        print(f"Parsed {len(texts)} texts from CSV")
 
        if not texts:
            print("No valid texts found in CSV")
            return jsonify({"error": "No valid texts found in CSV"}), 400
        
        print(f"Starting prediction with {model_type} model...")
        
        # Predict using the selected model
        results = []
        for text in texts:
            if model_type == 'bert-emotions':
                emotion, feeling, confidence = predict_bert_emotions(text)
                results.append({
                    "text": text,
                    "emotion": emotion or "AUSENTE",
                    "feeling": None,  # BERTimbau Emoções doesn't return feelings
                    "confidence": float(confidence) if confidence else 0.0
                })
            elif model_type == 'bert-sentiment':
                emotion, feeling, confidence = predict_bert_sentiment(text)
                results.append({
                    "text": text,
                    "emotion": emotion or "AUSENTE",
                    "feeling": feeling or "Neutro",
                    "confidence": float(confidence) if confidence else 0.0
                })
            elif model_type == 'mnb':
                emotion, feeling, confidence = predict_mnb(text)
                results.append({
                    "text": text,
                    "emotion": emotion or "AUSENTE",
                    "feeling": feeling or "Neutro",
                    "confidence": float(confidence) if confidence else 0.0
                })
            else:
                return jsonify({"error": f"Unknown model type: {model_type}"}), 400
        
        print(f"Prediction completed for {len(results)} texts")
        
        # Save results to file for logging
        import json
        from datetime import datetime
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_texts": len(texts),
            "model": model_type.upper(),
            "results": results
        }
        
        # Save to JSON file
        log_filename = f"emotion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_path = os.path.join(API_DIR, log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved to: {log_filename}")
        
        return jsonify({
            "results": results,
            "total_texts": len(texts),
            "model": model_type.upper(),
            "log_file": log_filename
        })
        
    except Exception as e:
        print(f"Error in predict_csv: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def load_metrics_from_file():
    """Load metrics from JSON file if it exists, otherwise return None"""
    metrics_file = os.path.join(API_DIR, 'model_metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics from {metrics_file}: {e}")
            return None
    return None

def get_default_metrics():
    """Get default metrics - fixed values from evaluation results"""
    return {
        'bert-emotions': {
            'model_name': 'BERTimbau Emoções',
            'accuracy': 0.56,
            'f1_macro': 0.33,
            'precision': 0.37,
            'recall': 0.31,
            'description': 'Modelo treinado para classificar 7 emoções'
        },
        'bert-sentiment': {
            'model_name': 'BERTimbau Sentimentos',
            'accuracy': 0.96,
            'f1_macro': 0.94,
            'precision': 0.95,
            'recall': 0.93,
            'description': 'Modelo treinado para classificar 3 sentimentos: NEGATIVO, NEUTRO, POSITIVO'
        },
        'mnb': {
            'model_name': 'MNB Sentimentos',
            'accuracy': 0.75,
            'f1_macro': 0.68,
            'precision': 0.71,
            'recall': 0.65,
            'description': 'Modelo Multinomial Naive Bayes treinado para classificar sentimentos: NEGATIVO, NEUTRO, POSITIVO'
        }
    }

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics for the selected model
    
    Metrics are loaded from model_metrics.json if it exists, otherwise default values are used.
    To update metrics, create/update api/model_metrics.json with the latest evaluation results.
    """
    model_type = request.args.get('model', 'bert-emotions').lower()
    
    # Try to load metrics from file first
    saved_metrics = load_metrics_from_file()
    
    if saved_metrics and model_type in saved_metrics:
        # Use saved metrics if available
        metrics = saved_metrics[model_type]
        print(f"Loaded metrics for {model_type} from file")
    else:
        # Use default metrics
        default_metrics = get_default_metrics()
        if model_type not in default_metrics:
            return jsonify({"error": f"Unknown model type: {model_type}"}), 400
        metrics = default_metrics[model_type]
        print(f"Using default metrics for {model_type}")
    
    return jsonify(metrics)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        model_type = data.get('model', 'bert-emotions').lower()
        
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        # Predict using the selected model
        results = []
        for text in texts:
            if model_type == 'bert-emotions':
                emotion, feeling, confidence = predict_bert_emotions(text)
            elif model_type == 'bert-sentiment':
                emotion, feeling, confidence = predict_bert_sentiment(text)
            elif model_type == 'mnb':
                emotion, feeling, confidence = predict_mnb(text)
            else:
                return jsonify({"error": f"Unknown model type: {model_type}"}), 400
            
            results.append({
                "text": text,
                "emotion": emotion or "AUSENTE",
                "feeling": feeling or "Neutro",
                "confidence": float(confidence) if confidence else 0.0
            })
        
        return jsonify({
            "results": results,
            "model": model_type.upper(),
            "total_texts": len(texts)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Railway, Render, etc. provide this)
    port = int(os.getenv('PORT', 5000))
    
    print("\n" + "="*60)
    print(f"API server starting on http://0.0.0.0:{port}")
    print("="*60)
    print("\nAvailable models:")
    print(f"  - bert-emotions: {'✅' if bert_emotions_model else '❌'}")
    print(f"  - bert-sentiment: {'✅' if bert_sentiment_model else '❌'}")
    print(f"  - mnb: {'✅' if mnb_model else '❌'}")
    print("\n" + "="*60)
    
    # Disable debug mode in production (set by platforms)
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        print(f"Error starting on port {port}: {e}")
        # Fallback to default port
        app.run(host='0.0.0.0', port=5000, debug=debug_mode)
