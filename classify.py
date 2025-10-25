"""
Inference Pipeline for AI/Real Image Classification
"""
import os
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import argparse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import config
from train import ClassifierTrainer

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self, model_path: str = None, clip_model_name: str = config.CLIP_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Load classifier
        self.classifier = None
        self.scaler = None
        self.training_history = None
        
        if model_path and os.path.exists(model_path):
            self.load_classifier(model_path)
        else:
            logger.warning("No classifier model loaded. Please train a model first.")
    
    def load_classifier(self, model_path: str):
        """Load trained classifier and scaler"""
        model_path = Path(model_path)
        
        try:
            # Load classifier
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load scaler
            scaler_path = model_path.with_suffix('.scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load training history
            history_path = model_path.with_suffix('.history.pkl')
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            
            logger.info(f"Classifier loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for CLIP model"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use CLIP processor for consistent preprocessing
            inputs = self.clip_processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        try:
            pixel_values = self.preprocess_image(image_path)
            
            with torch.no_grad():
                # Get image features
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            raise
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify a single image"""
        if self.classifier is None or self.scaler is None:
            raise ValueError("Classifier not loaded. Please load a trained model first.")
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(image_path)
            
            # Scale features
            embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
            
            # Predict
            prediction = self.classifier.predict(embedding_scaled)[0]
            probability = self.classifier.predict_proba(embedding_scaled)[0]
            
            # Map prediction to label
            label = "AI" if prediction == 0 else "Real"
            confidence = max(probability)
            
            result = {
                'image_path': image_path,
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'AI': float(probability[0]),
                    'Real': float(probability[1])
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            raise
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.classify_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def classify_from_url(self, image_url: str) -> Dict[str, Any]:
        """Classify image from URL"""
        import requests
        from io import BytesIO
        
        try:
            # Download image
            response = requests.get(image_url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Load image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Process with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten()
            
            # Scale and predict
            embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
            prediction = self.classifier.predict(embedding_scaled)[0]
            probability = self.classifier.predict_proba(embedding_scaled)[0]
            
            label = "AI" if prediction == 0 else "Real"
            confidence = max(probability)
            
            return {
                'image_url': image_url,
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'AI': float(probability[0]),
                    'Real': float(probability[1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying image from URL {image_url}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.training_history is None:
            return {"error": "No training history available"}
        
        return {
            'classifier_type': self.training_history.get('classifier_type', 'unknown'),
            'test_accuracy': self.training_history.get('test_accuracy', 0),
            'cv_mean': self.training_history.get('cv_mean', 0),
            'cv_std': self.training_history.get('cv_std', 0),
            'n_features': self.training_history.get('n_features', 0),
            'n_samples': self.training_history.get('n_samples', 0),
            'classification_report': self.training_history.get('classification_report', {})
        }

def main():
    parser = argparse.ArgumentParser(description="Classify images as AI-generated or real")
    parser.add_argument("--image", type=str, help="Path to image file to classify")
    parser.add_argument("--batch", type=str, nargs="+", help="Multiple image paths to classify")
    parser.add_argument("--url", type=str, help="URL of image to classify")
    parser.add_argument("--model", type=str, default=os.path.join(config.MODELS_DIR, "classifier.pkl"), help="Path to trained model")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ImageClassifier(args.model)
    
    results = []
    
    # Classify single image
    if args.image:
        result = classifier.classify_image(args.image)
        results.append(result)
        print(f"Image: {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: AI={result['probabilities']['AI']:.4f}, Real={result['probabilities']['Real']:.4f}")
    
    # Classify batch
    if args.batch:
        batch_results = classifier.classify_batch(args.batch)
        results.extend(batch_results)
        
        print(f"\nBatch classification results:")
        for result in batch_results:
            if 'error' in result:
                print(f"Error classifying {result['image_path']}: {result['error']}")
            else:
                print(f"{result['image_path']}: {result['prediction']} ({result['confidence']:.4f})")
    
    # Classify from URL
    if args.url:
        result = classifier.classify_from_url(args.url)
        results.append(result)
        print(f"\nURL: {args.url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    # Save results
    if args.output and results:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Show model info
    if not args.image and not args.batch and not args.url:
        print("Model Information:")
        info = classifier.get_model_info()
        for key, value in info.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()
