from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import json
import pickle
import re
import io
import pandas as pd
import requests
import shutil
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
    # Local development or Railway
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
BERT_EMOTIONS_PATH = os.path.join(MODELS_DIR, "bertimbau-mlp-ai")
BERT_SENTIMENT_PATH = os.path.join(MODELS_DIR, "bertimbau-mlp-sentiment")
MNB_MODEL_DIR = os.path.join(MODELS_DIR, "mnb")
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Model download URLs (set via environment variables)
BERT_EMOTIONS_MODEL_URL = os.getenv('BERT_EMOTIONS_MODEL_URL', '')
BERT_SENTIMENT_MODEL_URL = os.getenv('BERT_SENTIMENT_MODEL_URL', '')

def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer instead of actual binary"""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except:
        return False

def convert_google_drive_link(share_url):
    """Convert Google Drive share link to direct download link"""
    # Google Drive share links come in different formats:
    # Format 1: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    # Format 2: https://drive.google.com/open?id=FILE_ID
    # Format 3: https://drive.google.com/uc?id=FILE_ID (already direct)
    
    import re
    
    # Extract file ID from various Google Drive URL formats
    file_id = None
    
    # Try to extract from /d/FILE_ID/ pattern
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', share_url)
    if match:
        file_id = match.group(1)
    else:
        # Try to extract from ?id=FILE_ID pattern
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', share_url)
        if match:
            file_id = match.group(1)
    
    if file_id:
        # Convert to direct download URL
        # For large files, Google Drive may show a virus scan warning
        # Use uc?export=download&id= format, and handle virus scan confirmation
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return direct_url
    else:
        # If we can't extract ID, return original URL (might already be direct)
        print(f"   ⚠️  Could not extract file ID from Google Drive URL, using as-is")
        return share_url

def download_from_git_lfs(model_path, model_name):
    """Try to download model from Git LFS using git lfs pull and store in volume"""
    if not is_lfs_pointer(model_path):
        return False  # File is already a binary, not a pointer
    
    print(f"📥 Detected Git LFS pointer for {model_name}")
    print(f"   Note: Railway doesn't include .git in Docker build, so Git LFS may not work")
    print(f"   Will try Git LFS, but URL fallback is recommended for Railway")
    print(f"   Target location: {model_path}")
    try:
        import subprocess
        
        # Check if .git directory exists
        # Railway doesn't include .git in Docker build context, so it won't be available
        # We'll try git lfs pull anyway - it might work if Railway provides Git access
        git_dir = os.path.join(BASE_DIR, '.git')
        work_dir = BASE_DIR
        
        if not os.path.exists(git_dir):
            # Try parent directory (in case repo root is different)
            parent_git = os.path.join(os.path.dirname(BASE_DIR), '.git')
            if os.path.exists(parent_git):
                git_dir = parent_git
                work_dir = os.path.dirname(BASE_DIR)
                print(f"   Found .git in parent directory: {git_dir}")
            else:
                print(f"⚠️  .git directory not found - Railway doesn't include it in Docker build")
                print(f"   Will try git lfs pull anyway (may fail, will use URL fallback if needed)")
                work_dir = BASE_DIR  # Try anyway
        
        # Ensure the target directory exists (create in volume)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Get relative path from BASE_DIR for git lfs pull
        # BASE_DIR is /app, model_path is /app/models/bertimbau-mlp-ai/bertimbau_mlp.pt
        # Relative path: models/bertimbau-mlp-ai/bertimbau_mlp.pt
        # But Git tracks files as app/models/... (relative to repo root)
        # Since we copied .git/ to /app/.git/, the repo root is /app/
        # So we need to use "app/models/..." path for git lfs pull
        rel_path_from_base = os.path.relpath(model_path, BASE_DIR)
        # Try both paths: the relative path and with "app/" prefix
        # Git might track files as "app/models/..." even though repo root is /app/
        git_paths_to_try = [
            f"app/{rel_path_from_base}",  # app/models/bertimbau-mlp-ai/bertimbau_mlp.pt
            rel_path_from_base,  # models/bertimbau-mlp-ai/bertimbau_mlp.pt
        ]
        
        print(f"   Model path: {model_path}")
        print(f"   Relative from BASE_DIR: {rel_path_from_base}")
        print(f"   Trying Git paths: {git_paths_to_try}")
        
        # Try git lfs pull with different path formats
        # This downloads to the volume location since volume is mounted at runtime
        # Note: On Railway, we might not have .git directory, but git lfs might still work
        result = None
        success = False
        
        # Determine working directory for git commands
        work_dir = BASE_DIR
        if os.path.exists(git_dir):
            work_dir = BASE_DIR
        else:
            # Try to find where git is available
            work_dir = os.getcwd()
        
        for git_path in git_paths_to_try:
            print(f"   Trying: git lfs pull --include {git_path}")
            result = subprocess.run(
                ['git', 'lfs', 'pull', '--include', git_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print(f"   ✅ Success with path: {git_path}")
                success = True
                break
            else:
                print(f"   ⚠️  Failed with path: {git_path} (exit code: {result.returncode})")
        
        # If specific paths failed, try pulling all LFS files
        if not success:
            print(f"   Trying: git lfs pull (all files)")
            result = subprocess.run(
                ['git', 'lfs', 'pull'],  # Try pulling all LFS files
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                success = True
        
        if success and result and result.returncode == 0:
            # Verify the file is now a binary (not a pointer)
            if os.path.exists(model_path) and not is_lfs_pointer(model_path):
                file_size = os.path.getsize(model_path) / (1024*1024)
                print(f"✅ {model_name} fetched from Git LFS successfully!")
                print(f"   File size: {file_size:.2f} MB (stored in Railway volume)")
                return True
            else:
                print(f"⚠️  File still appears to be a pointer after git lfs pull")
                if result.stdout:
                    print(f"   Git LFS output: {result.stdout}")
                return False
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"⚠️  Git LFS pull failed (exit code {result.returncode})")
            print(f"   Error: {error_msg}")
            
            # Check if it's an authentication issue
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print(f"   💡 Tip: Railway may need Git credentials for private repos")
                print(f"   💡 Consider using BERT_EMOTIONS_MODEL_URL and BERT_SENTIMENT_MODEL_URL env vars instead")
            
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Git LFS pull timed out after 600 seconds")
        return False
    except Exception as e:
        print(f"❌ Error fetching from Git LFS: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_model_if_needed(model_path, model_url, model_name):
    """Download model if it doesn't exist and URL is provided"""
    print(f"\n🔍 Checking {model_name}...")
    print(f"   Path: {model_path}")
    print(f"   Exists: {os.path.exists(model_path)}")
    print(f"   URL provided: {bool(model_url)}")
    
    # First, check if it's a Git LFS pointer
    if is_lfs_pointer(model_path):
        print(f"⚠️  {model_name} is a Git LFS pointer")
        
        # On Railway, Git LFS usually won't work (no .git directory)
        # So prioritize URL download if available
        if model_url:
            print(f"   Using URL download (recommended for Railway)...")
            os.remove(model_path)  # Remove pointer file
        else:
            # Try Git LFS as fallback (might work in some environments)
            print(f"   No URL provided, trying Git LFS...")
            if download_from_git_lfs(model_path, model_name):
                return True
            print(f"   ⚠️  Git LFS failed. Please set {model_name.upper().replace(' ', '_')}_MODEL_URL environment variable")
            return False
    
    # If file exists and is not a pointer, we're good
    if os.path.exists(model_path) and not is_lfs_pointer(model_path):
        print(f"   ✅ File already exists, skipping download")
        return True
    
    # Download from URL if provided
    # Check if URL is provided (not empty string) and file doesn't exist
    if model_url and model_url.strip() and not os.path.exists(model_path):
        print(f"📥 Downloading {model_name} from {model_url}...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Convert Google Drive share link to direct download link if needed
            download_url = model_url
            if "drive.google.com" in model_url:
                print(f"   Detected Google Drive link, converting to direct download...")
                download_url = convert_google_drive_link(model_url)
                print(f"   Using direct download URL")
            
            # For Google Drive, handle virus scan warning for large files
            session = requests.Session()
            response = session.get(download_url, stream=True, timeout=600, allow_redirects=True)
            
            # Check if Google Drive is showing a virus scan warning page
            # Large files (>100MB) trigger a warning that needs confirmation
            if 'virus scan warning' in response.text.lower() or 'download anyway' in response.text.lower():
                print(f"   ⚠️  Google Drive virus scan warning detected, attempting to bypass...")
                # Extract the confirmation link from the warning page
                import re
                confirm_match = re.search(r'href="(/uc\?export=download[^"]*)"', response.text)
                if confirm_match:
                    confirm_url = "https://drive.google.com" + confirm_match.group(1)
                    print(f"   Using confirmation link...")
                    response = session.get(confirm_url, stream=True, timeout=600, allow_redirects=True)
                else:
                    # Try alternative method: use confirm parameter
                    if 'id=' in download_url:
                        file_id = re.search(r'id=([^&]+)', download_url).group(1)
                        confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                        print(f"   Trying alternative download method...")
                        response = session.get(confirm_url, stream=True, timeout=600, allow_redirects=True)
            
            response.raise_for_status()
            
            # Check if we got HTML instead of binary (still showing warning page)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type and len(response.content) < 10000:
                # Likely still a warning page, try one more time with confirm=t
                if 'drive.google.com' in download_url and 'id=' in download_url:
                    file_id = re.search(r'id=([^&]+)', download_url).group(1)
                    final_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                    print(f"   Retrying with confirmation parameter...")
                    response = session.get(final_url, stream=True, timeout=600, allow_redirects=True)
                    response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='', flush=True)
                        else:
                            print(f"\r   Progress: {downloaded / (1024*1024):.1f} MB downloaded", end='', flush=True)
            
            print(f"\n✅ {model_name} downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            return False
    
    # If we get here, file doesn't exist
    if not model_url or not model_url.strip():
        print(f"   ⚠️  No URL provided and file doesn't exist!")
        print(f"   ⚠️  Set {model_name.upper().replace(' ', '_')}_MODEL_URL environment variable in Railway")
        print(f"   ⚠️  Current value: '{model_url}' (empty or not set)")
        return False
    else:
        print(f"   ⚠️  File doesn't exist but URL is set. Download should have happened above.")
        print(f"   ⚠️  This shouldn't happen - check the download logic above.")
    
    return os.path.exists(model_path) and not is_lfs_pointer(model_path)

# Copy src directories to volume if they don't exist (volume overlay hides build files)
print("Setting up model directories in volume...")
for model_dir, temp_path in [
    (BERT_EMOTIONS_PATH, "/tmp/models/bertimbau-mlp-ai"),
    (BERT_SENTIMENT_PATH, "/tmp/models/bertimbau-mlp-sentiment")
]:
    if not os.path.exists(os.path.join(model_dir, "src")):
        temp_src = os.path.join(temp_path, "src")
        if os.path.exists(temp_src):
            os.makedirs(model_dir, exist_ok=True)
            shutil.copytree(temp_src, os.path.join(model_dir, "src"), dirs_exist_ok=True)
            print(f"   ✅ Copied src directory to {model_dir}")

# Try to download models if they don't exist or are LFS pointers
bert_emotions_path = os.path.join(BERT_EMOTIONS_PATH, "bertimbau_mlp.pt")
bert_sentiment_path = os.path.join(BERT_SENTIMENT_PATH, "bertimbau_sentiment_best.pt")

# Try Git LFS first, then fallback to URL if provided
print("\n" + "="*60)
print("DOWNLOADING MODEL WEIGHTS")
print("="*60)
print(f"BERT_EMOTIONS_MODEL_URL: {'✅ SET' if BERT_EMOTIONS_MODEL_URL and BERT_EMOTIONS_MODEL_URL.strip() else '❌ NOT SET'}")
if BERT_EMOTIONS_MODEL_URL:
    print(f"   Value: {BERT_EMOTIONS_MODEL_URL[:50]}..." if len(BERT_EMOTIONS_MODEL_URL) > 50 else f"   Value: {BERT_EMOTIONS_MODEL_URL}")
print(f"BERT_SENTIMENT_MODEL_URL: {'✅ SET' if BERT_SENTIMENT_MODEL_URL and BERT_SENTIMENT_MODEL_URL.strip() else '❌ NOT SET'}")
if BERT_SENTIMENT_MODEL_URL:
    print(f"   Value: {BERT_SENTIMENT_MODEL_URL[:50]}..." if len(BERT_SENTIMENT_MODEL_URL) > 50 else f"   Value: {BERT_SENTIMENT_MODEL_URL}")
print(f"\nFile check:")
print(f"   {bert_emotions_path} exists: {os.path.exists(bert_emotions_path)}")
print(f"   {bert_sentiment_path} exists: {os.path.exists(bert_sentiment_path)}")
print("="*60 + "\n")

# Download models - this WILL attempt download if URLs are set
download_model_if_needed(bert_emotions_path, BERT_EMOTIONS_MODEL_URL, "BERTimbau Emoções")
download_model_if_needed(bert_sentiment_path, BERT_SENTIMENT_MODEL_URL, "BERTimbau Sentimentos")

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
        raise FileNotFoundError(f"Source path not found: {src_path} (should have been copied earlier)")
    
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
        # Verify file is not empty and is a valid binary file
        file_size = os.path.getsize(model_path)
        print(f"Loading weights from {model_path}...")
        print(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        # Try loading with explicit file mode
        try:
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
        except Exception as load_error:
            print(f"❌ Error loading model weights: {load_error}")
            print(f"   File exists: {os.path.exists(model_path)}")
            print(f"   File size: {file_size} bytes")
            # Check first few bytes to see if file is corrupted
            try:
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(10)
                    print(f"   First 10 bytes (hex): {first_bytes.hex()}")
            except:
                pass
            raise
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
        raise FileNotFoundError(f"Source path not found: {src_path} (should have been copied earlier)")
    
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
        # Verify file is not empty and is a valid binary file
        file_size = os.path.getsize(model_path)
        print(f"Loading weights from {model_path}...")
        print(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        # Try loading with explicit file mode
        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
            bert_sentiment_model.load_state_dict(state_dict)
            bert_sentiment_model.eval()
            print("✅ BERTimbau Sentimentos model loaded successfully!")
            print(f"   Using: bertimbau_sentiment_best.pt")
        except Exception as load_error:
            print(f"❌ Error loading model weights: {load_error}")
            print(f"   File exists: {os.path.exists(model_path)}")
            print(f"   File size: {file_size} bytes")
            # Check first few bytes to see if file is corrupted
            try:
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(10)
                    print(f"   First 10 bytes (hex): {first_bytes.hex()}")
            except:
                pass
            raise
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
    # Note: .pkl files are NOT in Git LFS, so they're copied normally during Docker build
    # However, if Railway volume is mounted at /app/models, we need to ensure they're available
    mnb_model_path = os.path.join(MNB_MODEL_DIR, "mnb_model.pkl")
    mnb_vectorizer_path = os.path.join(MNB_MODEL_DIR, "mnb_vectorizer.pkl")
    mnb_mlb_path = os.path.join(MNB_MODEL_DIR, "mnb_mlb.pkl")
    
    # Check if files exist, if not, try to copy from temp location to volume
    # (Volume overlays /app/models, so build files are hidden - we copied them to /tmp during build)
    if not all(os.path.exists(p) for p in [mnb_model_path, mnb_vectorizer_path, mnb_mlb_path]):
        print(f"⚠️  Some MNB .pkl files not found in volume, copying from temp location...")
        # .pkl files were copied to /tmp/models/mnb during Docker build
        # (because volume mount overlays /app/models, hiding build files)
        temp_models_dir = "/tmp/models/mnb"
        if os.path.exists(temp_models_dir):
            print(f"   Found .pkl files in temp location, copying to volume...")
            os.makedirs(MNB_MODEL_DIR, exist_ok=True)
            for pkl_file in ["mnb_model.pkl", "mnb_vectorizer.pkl", "mnb_mlb.pkl"]:
                src = os.path.join(temp_models_dir, pkl_file)
                dst = os.path.join(MNB_MODEL_DIR, pkl_file)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"   ✅ Copied {pkl_file} to volume ({os.path.getsize(dst) / 1024:.1f} KB)")
                elif os.path.exists(src):
                    print(f"   ℹ️  {pkl_file} already exists in volume")
        else:
            print(f"   ⚠️  Temp location {temp_models_dir} not found")
    
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
        print(f"   Note: .pkl files are in regular Git (not LFS), so they should be copied during build")
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
    """Health check endpoint - Railway uses this to verify deployment is ready"""
    models_loaded = {
        "bert_emotions": bert_emotions_model is not None,
        "bert_sentiment": bert_sentiment_model is not None,
        "mnb": mnb_model is not None
    }
    
    # Check if all critical models are loaded
    all_loaded = all(models_loaded.values())
    
    status_code = 200 if all_loaded else 503  # 503 = Service Unavailable
    status = "ready" if all_loaded else "loading"
    
    return jsonify({
        "status": status,
        "models": models_loaded,
        "ready": all_loaded
    }), status_code

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
