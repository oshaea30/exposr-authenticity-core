"""
Configuration settings for Exposr Authenticity Core
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data paths
DATA_DIR = "data"
AI_IMAGES_DIR = os.path.join(DATA_DIR, "ai")
REAL_IMAGES_DIR = os.path.join(DATA_DIR, "real")
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"

# Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLASSIFIER_TYPE = "random_forest"  # Options: "logistic_regression", "random_forest", "svm"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training settings
TRAINING_FREQUENCY = os.getenv("TRAINING_FREQUENCY", "weekly")  # "daily" or "weekly"
TRAINING_TIME = os.getenv("TRAINING_TIME", "02:00")  # 24-hour format
MIN_IMAGES_PER_CLASS = 100
MAX_IMAGES_PER_CLASS = 5000

# Scraping settings
SCRAPE_DELAY_MIN = 1  # Minimum delay between requests (seconds)
SCRAPE_DELAY_MAX = 3  # Maximum delay between requests (seconds)
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# File settings
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
EMBEDDING_DIMENSION = 512  # CLIP ViT-B/32 embedding dimension

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
for directory in [DATA_DIR, AI_IMAGES_DIR, REAL_IMAGES_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
