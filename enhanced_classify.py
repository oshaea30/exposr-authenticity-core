"""
Enhanced Inference Pipeline for Exposr Integration
"""
import os
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import argparse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from datetime import datetime
import config
from enhanced_train import EnhancedClassifierTrainer

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EnhancedImageClassifier:
    def __init__(self, model_path: str = None, clip_model_name: str = config.CLIP_MODEL_NAME, version: str = "1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = version
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
        self.model_metadata = None
        self.inference_history = []
        
        if model_path and os.path.exists(model_path):
            self.load_classifier(model_path)
        else:
            logger.warning("No classifier model loaded. Please train a model first.")
    
    def load_classifier(self, model_path: str):
        """Load trained classifier with metadata"""
        model_path = Path(model_path)
        
        try:
            # Load classifier
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load scaler
            scaler_path = model_path.with_suffix('.scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model metadata
            history_path = model_path.with_suffix('.history.json')
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                self.model_metadata = {}
            
            logger.info(f"Classifier loaded from {model_path}")
            logger.info(f"Model version: {self.model_metadata.get('version', 'unknown')}")
            logger.info(f"Model accuracy: {self.model_metadata.get('performance', {}).get('test_accuracy', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Standardized image preprocessing matching training pipeline"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use CLIP processor for consistent preprocessing
            inputs = self.clip_processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding matching training pipeline"""
        try:
            pixel_values = self.preprocess_image(image_path)
            
            with torch.no_grad():
                # Get image features
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                
                # Normalize features (matching training pipeline)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            raise
    
    def classify_image(self, image_path: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Enhanced image classification with detailed results"""
        if self.classifier is None or self.scaler is None:
            raise ValueError("Classifier not loaded. Please load a trained model first.")
        
        start_time = datetime.now()
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(image_path)
            
            # Scale features (matching training pipeline)
            embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
            
            # Predict
            prediction = self.classifier.predict(embedding_scaled)[0]
            probability = self.classifier.predict_proba(embedding_scaled)[0]
            
            # Map prediction to label
            label = "AI" if prediction == 0 else "Real"
            confidence = max(probability)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'image_path': image_path,
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'AI': float(probability[0]),
                    'Real': float(probability[1])
                },
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': start_time.isoformat(),
                'model_version': self.version,
                'model_metadata': self.model_metadata if include_metadata else None
            }
            
            # Log inference
            self.log_inference(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            raise
    
    def classify_batch(self, image_paths: List[str], include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Enhanced batch classification with progress tracking"""
        results = []
        total_images = len(image_paths)
        
        logger.info(f"Starting batch classification of {total_images} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.classify_image(image_path, include_metadata)
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0 or i == total_images - 1:
                    logger.info(f"Processed {i + 1}/{total_images} images")
                    
            except Exception as e:
                logger.error(f"Failed to classify {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Batch classification completed: {len(results)} results")
        return results
    
    def classify_from_url(self, image_url: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Enhanced URL classification with better error handling"""
        import requests
        from io import BytesIO
        
        start_time = datetime.now()
        
        try:
            # Download image with timeout and size limits
            response = requests.get(image_url, timeout=config.REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > config.MAX_IMAGE_SIZE:
                raise ValueError(f"Image too large: {content_length} bytes")
            
            # Load image
            image_data = BytesIO(response.content)
            image = Image.open(image_data).convert("RGB")
            
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
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'image_url': image_url,
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'AI': float(probability[0]),
                    'Real': float(probability[1])
                },
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': start_time.isoformat(),
                'model_version': self.version,
                'model_metadata': self.model_metadata if include_metadata else None
            }
            
            # Log inference
            self.log_inference(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image from URL {image_url}: {e}")
            raise
    
    def log_inference(self, result: Dict[str, Any]):
        """Log inference for analytics"""
        self.inference_history.append(result)
        
        # Keep only last 1000 inferences to prevent memory issues
        if len(self.inference_history) > 1000:
            self.inference_history = self.inference_history[-1000:]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.model_metadata is None:
            return {"error": "No model metadata available"}
        
        return {
            'model_version': self.version,
            'classifier_type': self.model_metadata.get('classifier_type', 'unknown'),
            'training_timestamp': self.model_metadata.get('timestamp', 'unknown'),
            'performance': self.model_metadata.get('performance', {}),
            'best_params': self.model_metadata.get('best_params', {}),
            'data_metadata': self.model_metadata.get('data_metadata', {}),
            'device': str(self.device),
            'clip_model': config.CLIP_MODEL_NAME,
            'total_inferences': len(self.inference_history)
        }
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        if not self.inference_history:
            return {"total_inferences": 0}
        
        # Calculate statistics
        total_inferences = len(self.inference_history)
        avg_processing_time = np.mean([r.get('processing_time_ms', 0) for r in self.inference_history])
        
        # Prediction distribution
        predictions = [r.get('prediction', 'Unknown') for r in self.inference_history]
        ai_count = predictions.count('AI')
        real_count = predictions.count('Real')
        
        # Confidence statistics
        confidences = [r.get('confidence', 0) for r in self.inference_history if 'confidence' in r]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'total_inferences': total_inferences,
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'prediction_distribution': {
                'AI': ai_count,
                'Real': real_count
            },
            'avg_confidence': round(avg_confidence, 4),
            'last_inference': self.inference_history[-1].get('timestamp') if self.inference_history else None
        }
    
    def save_inference_history(self, filepath: str = None):
        """Save inference history for analysis"""
        if not filepath:
            filepath = f"inference_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.inference_history, f, indent=2)
        
        logger.info(f"Inference history saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Image Classification for Exposr")
    parser.add_argument("--image", type=str, help="Path to image file to classify")
    parser.add_argument("--batch", type=str, nargs="+", help="Multiple image paths to classify")
    parser.add_argument("--url", type=str, help="URL of image to classify")
    parser.add_argument("--model", type=str, default="models/classifier.pkl", help="Path to trained model")
    parser.add_argument("--version", type=str, default="1.0", help="Model version")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--stats", action="store_true", help="Show inference statistics")
    parser.add_argument("--info", action="store_true", help="Show model information")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = EnhancedImageClassifier(args.model, version=args.version)
    
    results = []
    
    # Classify single image
    if args.image:
        result = classifier.classify_image(args.image)
        results.append(result)
        print(f"🖼️  Image: {args.image}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
        print(f"📈 Probabilities: AI={result['probabilities']['AI']:.4f}, Real={result['probabilities']['Real']:.4f}")
    
    # Classify batch
    if args.batch:
        batch_results = classifier.classify_batch(args.batch)
        results.extend(batch_results)
        
        print(f"\n📦 Batch Classification Results:")
        print("-" * 50)
        for result in batch_results:
            if 'error' in result:
                print(f"❌ {result['image_path']}: {result['error']}")
            else:
                print(f"✅ {result['image_path']}: {result['prediction']} ({result['confidence']:.4f})")
    
    # Classify from URL
    if args.url:
        result = classifier.classify_from_url(args.url)
        results.append(result)
        print(f"\n🌐 URL: {args.url}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
    
    # Show model info
    if args.info:
        print(f"\n📋 Model Information:")
        info = classifier.get_model_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    # Show inference stats
    if args.stats:
        print(f"\n📊 Inference Statistics:")
        stats = classifier.get_inference_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    # Save results
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Default: show model info if no other action
    if not args.image and not args.batch and not args.url and not args.info and not args.stats:
        print("🤖 Enhanced Image Classifier for Exposr")
        print("Use --help to see available options")
        print("\nQuick examples:")
        print("  python enhanced_classify.py --image path/to/image.jpg")
        print("  python enhanced_classify.py --url https://example.com/image.jpg")
        print("  python enhanced_classify.py --info")
        print("  python enhanced_classify.py --stats")

if __name__ == "__main__":
    main()
