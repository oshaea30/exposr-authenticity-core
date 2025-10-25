"""
Test script for Exposr Authenticity Core
"""
import os
import sys
import logging
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import torch
        import torchvision
        import transformers
        import sklearn
        import numpy as np
        import PIL
        import requests
        import flask
        import schedule
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_directories():
    """Test if all required directories exist"""
    logger.info("Testing directories...")
    
    required_dirs = [
        config.DATA_DIR,
        config.AI_IMAGES_DIR,
        config.REAL_IMAGES_DIR,
        config.EMBEDDINGS_DIR,
        config.MODELS_DIR
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"❌ Directory missing: {dir_path}")
            return False
    
    logger.info("✅ All directories exist")
    return True

def test_clip_model():
    """Test if CLIP model can be loaded"""
    logger.info("Testing CLIP model loading...")
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        
        logger.info("✅ CLIP model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ CLIP model loading failed: {e}")
        return False

def test_classifier_creation():
    """Test if classifier can be created"""
    logger.info("Testing classifier creation...")
    
    try:
        from train import ClassifierTrainer
        
        trainer = ClassifierTrainer()
        classifier = trainer.create_classifier()
        
        logger.info("✅ Classifier created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Classifier creation failed: {e}")
        return False

def test_image_classifier():
    """Test if ImageClassifier can be initialized"""
    logger.info("Testing ImageClassifier initialization...")
    
    try:
        from classify import ImageClassifier
        
        # Initialize without model (should not fail)
        classifier = ImageClassifier()
        
        logger.info("✅ ImageClassifier initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ ImageClassifier initialization failed: {e}")
        return False

def test_scheduler():
    """Test if scheduler can be initialized"""
    logger.info("Testing scheduler initialization...")
    
    try:
        from scheduler import TrainingScheduler
        
        scheduler = TrainingScheduler()
        
        logger.info("✅ Scheduler initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Scheduler initialization failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be created"""
    logger.info("Testing Flask app creation...")
    
    try:
        from app import app
        
        # Test if app is created
        assert app is not None
        
        logger.info("✅ Flask app created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Flask app creation failed: {e}")
        return False

def test_config():
    """Test configuration values"""
    logger.info("Testing configuration...")
    
    required_configs = [
        'CLIP_MODEL_NAME',
        'CLASSIFIER_TYPE',
        'TRAINING_FREQUENCY',
        'TRAINING_TIME',
        'MIN_IMAGES_PER_CLASS',
        'MAX_IMAGES_PER_CLASS'
    ]
    
    for config_name in required_configs:
        if not hasattr(config, config_name):
            logger.error(f"❌ Missing config: {config_name}")
            return False
    
    logger.info("✅ Configuration is valid")
    return True

def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Running Exposr Authenticity Core tests...")
    
    tests = [
        test_imports,
        test_directories,
        test_config,
        test_clip_model,
        test_classifier_creation,
        test_image_classifier,
        test_scheduler,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
