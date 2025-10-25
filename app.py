"""
REST API for AI/Real Image Classification
"""
import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import config
from classify import ImageClassifier
from train import ClassifierTrainer
from scheduler import TrainingScheduler

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global classifier instance
classifier = None
scheduler = None

def initialize_classifier():
    """Initialize the classifier"""
    global classifier
    model_path = os.path.join(config.MODELS_DIR, "classifier.pkl")
    
    if os.path.exists(model_path):
        classifier = ImageClassifier(model_path)
        logger.info("Classifier loaded successfully")
    else:
        logger.warning("No trained model found. Please train a model first.")
        classifier = None

def initialize_scheduler():
    """Initialize the training scheduler"""
    global scheduler
    scheduler = TrainingScheduler()
    logger.info("Training scheduler initialized")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier is not None,
        'timestamp': str(datetime.now())
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify a single image"""
    if classifier is None:
        return jsonify({'error': 'Classifier not loaded. Please train a model first.'}), 500
    
    try:
        # Check if file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            # Classify image
            result = classifier.classify_image(tmp_file.name)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify_url', methods=['POST'])
def classify_image_url():
    """Classify an image from URL"""
    if classifier is None:
        return jsonify({'error': 'Classifier not loaded. Please train a model first.'}), 500
    
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url']
        result = classifier.classify_from_url(url)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error classifying image from URL: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    """Classify multiple images"""
    if classifier is None:
        return jsonify({'error': 'Classifier not loaded. Please train a model first.'}), 500
    
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No image files provided'}), 400
        
        results = []
        temp_files = []
        
        # Save all files temporarily
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
            file.save(tmp_file.name)
            temp_files.append(tmp_file.name)
        
        # Classify all images
        results = classifier.classify_batch(temp_files)
        
        # Clean up temporary files
        for tmp_file in temp_files:
            os.unlink(tmp_file)
        
        return jsonify({
            'results': results,
            'total_images': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics"""
    if classifier is None:
        return jsonify({'error': 'Classifier not loaded. Please train a model first.'}), 500
    
    try:
        info = classifier.get_model_info()
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Trigger manual model retraining"""
    try:
        data = request.get_json() or {}
        
        # Check if we should scrape new data
        scrape_data = data.get('scrape_data', True)
        
        if scheduler is None:
            scheduler = TrainingScheduler()
        
        # Run training pipeline
        scheduler.run_training_pipeline()
        
        # Reload classifier
        initialize_classifier()
        
        return jsonify({
            'message': 'Model retraining completed',
            'classifier_loaded': classifier is not None
        })
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training_history', methods=['GET'])
def get_training_history():
    """Get training history"""
    if scheduler is None:
        return jsonify({'error': 'Scheduler not initialized'}), 500
    
    try:
        history = scheduler.get_training_history()
        return jsonify({
            'history': history,
            'total_events': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'classifier_type': config.CLASSIFIER_TYPE,
        'training_frequency': config.TRAINING_FREQUENCY,
        'training_time': config.TRAINING_TIME,
        'min_images_per_class': config.MIN_IMAGES_PER_CLASS,
        'max_images_per_class': config.MAX_IMAGES_PER_CLASS,
        'clip_model_name': config.CLIP_MODEL_NAME,
        'supported_formats': config.SUPPORTED_IMAGE_FORMATS
    })

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Update config values
        if 'classifier_type' in data:
            config.CLASSIFIER_TYPE = data['classifier_type']
        if 'training_frequency' in data:
            config.TRAINING_FREQUENCY = data['training_frequency']
        if 'training_time' in data:
            config.TRAINING_TIME = data['training_time']
        if 'min_images_per_class' in data:
            config.MIN_IMAGES_PER_CLASS = data['min_images_per_class']
        if 'max_images_per_class' in data:
            config.MAX_IMAGES_PER_CLASS = data['max_images_per_class']
        
        return jsonify({'message': 'Configuration updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # Count images in directories
        ai_count = len(list(Path(config.AI_IMAGES_DIR).glob("*"))) if Path(config.AI_IMAGES_DIR).exists() else 0
        real_count = len(list(Path(config.REAL_IMAGES_DIR).glob("*"))) if Path(config.REAL_IMAGES_DIR).exists() else 0
        
        # Check if embeddings exist
        embeddings_path = Path(config.EMBEDDINGS_DIR) / "image_vectors.npy"
        embeddings_exist = embeddings_path.exists()
        
        # Check if model exists
        model_path = Path(config.MODELS_DIR) / "classifier.pkl"
        model_exists = model_path.exists()
        
        return jsonify({
            'ai_images': ai_count,
            'real_images': real_count,
            'total_images': ai_count + real_count,
            'embeddings_exist': embeddings_exist,
            'model_exists': model_exists,
            'classifier_loaded': classifier is not None,
            'scheduler_initialized': scheduler is not None
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request error"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the API server"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="REST API for AI/Real Image Classification")
    parser.add_argument("--host", type=str, default=config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Initialize components
    initialize_classifier()
    initialize_scheduler()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Set file upload limits
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_IMAGE_SIZE
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
