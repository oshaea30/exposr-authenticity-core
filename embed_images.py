"""
Standardized Embedding Pipeline with Version Tracking
"""
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle
import json
import hashlib
from datetime import datetime
import argparse
from transformers import CLIPProcessor, CLIPModel
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, model_name: str = config.CLIP_MODEL_NAME, version: str = "1.0"):
        self.model_name = model_name
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize pipeline metadata
        self.pipeline_metadata = {
            'version': version,
            'model_name': model_name,
            'device': str(self.device),
            'created_at': datetime.now().isoformat(),
            'embedding_dimension': 512,  # CLIP ViT-B/32
            'preprocessing': {
                'resize': 224,
                'normalize': True,
                'center_crop': True
            }
        }
        
        logger.info("Embedding pipeline initialized successfully")
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Standardized image preprocessing"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use CLIP processor for consistent preprocessing
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate standardized embedding for a single image"""
        try:
            pixel_values = self.preprocess_image(image_path)
            if pixel_values is None:
                return None
            
            with torch.no_grad():
                # Get image features
                image_features = self.model.get_image_features(pixel_values=pixel_values)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return None
    
    def generate_embeddings_batch(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for a batch of images with progress tracking"""
        embeddings = []
        total_images = len(image_paths)
        
        logger.info(f"Processing {total_images} images in batches of {batch_size}")
        
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Preprocess batch
            for path in batch_paths:
                pixel_values = self.preprocess_image(path)
                if pixel_values is not None:
                    batch_images.append(pixel_values)
            
            if not batch_images:
                continue
                
            try:
                # Stack batch
                batch_tensor = torch.cat(batch_images, dim=0)
                
                with torch.no_grad():
                    # Get image features
                    image_features = self.model.get_image_features(pixel_values=batch_tensor)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    batch_embeddings = image_features.cpu().numpy()
                    
                    for embedding in batch_embeddings:
                        embeddings.append(embedding.flatten())
                        
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Fallback to individual processing
                for path in batch_paths:
                    embedding = self.generate_embedding(path)
                    if embedding is not None:
                        embeddings.append(embedding)
            
            # Progress logging
            processed = min(i + batch_size, total_images)
            logger.info(f"Processed {processed}/{total_images} images ({processed/total_images*100:.1f}%)")
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def process_dataset(self, ai_dir: str, real_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Process entire dataset and generate embeddings with metadata"""
        logger.info("Starting standardized dataset processing")
        
        # Get image paths
        ai_paths = self.get_image_paths(ai_dir)
        real_paths = self.get_image_paths(real_dir)
        
        logger.info(f"Found {len(ai_paths)} AI images and {len(real_paths)} real images")
        
        # Generate embeddings
        logger.info("Generating AI image embeddings...")
        ai_embeddings = self.generate_embeddings_batch(ai_paths)
        
        logger.info("Generating real image embeddings...")
        real_embeddings = self.generate_embeddings_batch(real_paths)
        
        logger.info(f"Generated {len(ai_embeddings)} AI embeddings and {len(real_embeddings)} real embeddings")
        
        # Convert to numpy arrays
        ai_array = np.array(ai_embeddings) if ai_embeddings else np.array([])
        real_array = np.array(real_embeddings) if real_embeddings else np.array([])
        
        # Create labels
        labels = ['AI'] * len(ai_embeddings) + ['Real'] * len(real_embeddings)
        
        # Create processing metadata
        processing_metadata = {
            'pipeline_version': self.version,
            'model_name': self.model_name,
            'processing_timestamp': datetime.now().isoformat(),
            'ai_images_processed': len(ai_embeddings),
            'real_images_processed': len(real_embeddings),
            'total_embeddings': len(ai_embeddings) + len(real_embeddings),
            'embedding_dimension': ai_array.shape[1] if len(ai_array) > 0 else 0,
            'device_used': str(self.device),
            'ai_image_paths': ai_paths,
            'real_image_paths': real_paths
        }
        
        return ai_array, real_array, labels, processing_metadata
    
    def get_image_paths(self, directory: str) -> List[str]:
        """Get all valid image file paths from directory"""
        directory = Path(directory)
        image_paths = []
        
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        return [str(path) for path in image_paths]
    
    def save_embeddings_with_version(self, ai_embeddings: np.ndarray, real_embeddings: np.ndarray, 
                                   labels: List[str], processing_metadata: Dict[str, Any], 
                                   output_dir: str = config.EMBEDDINGS_DIR):
        """Save embeddings with version tracking"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versioned filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"embeddings_v{self.version}_{timestamp}.npy"
        output_path = output_dir / versioned_filename
        
        # Combine embeddings
        all_embeddings = np.vstack([ai_embeddings, real_embeddings]) if len(ai_embeddings) > 0 and len(real_embeddings) > 0 else np.array([])
        
        # Save embeddings
        np.save(output_path, all_embeddings)
        
        # Save metadata
        metadata = {
            'pipeline_metadata': self.pipeline_metadata,
            'processing_metadata': processing_metadata,
            'ai_count': len(ai_embeddings),
            'real_count': len(real_embeddings),
            'total_count': len(all_embeddings),
            'embedding_dimension': all_embeddings.shape[1] if len(all_embeddings) > 0 else 0,
            'labels': labels,
            'model_name': self.model_name,
            'version': self.version,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as latest
        latest_path = output_dir / "latest_embeddings.npy"
        latest_metadata_path = output_dir / "latest_embeddings.json"
        
        np.save(latest_path, all_embeddings)
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Embeddings saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Latest embeddings saved to {latest_path}")
        
        return output_path, metadata_path
    
    def load_embeddings_with_version(self, embeddings_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load embeddings with version information"""
        embeddings_path = Path(embeddings_path)
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        metadata_path = embeddings_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Fallback to old format
            metadata_path = embeddings_path.with_suffix('.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                metadata = {}
        
        return embeddings, metadata
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """Validate embedding quality"""
        if len(embeddings) == 0:
            logger.error("No embeddings found")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.error("Embeddings contain NaN or infinite values")
            return False
        
        # Check embedding dimension
        if embeddings.shape[1] != self.pipeline_metadata['embedding_dimension']:
            logger.error(f"Unexpected embedding dimension: {embeddings.shape[1]}")
            return False
        
        # Check for zero vectors
        zero_vectors = np.all(embeddings == 0, axis=1)
        if np.any(zero_vectors):
            logger.warning(f"Found {np.sum(zero_vectors)} zero vectors")
        
        logger.info("Embedding validation passed")
        return True
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            'pipeline_metadata': self.pipeline_metadata,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'processor_loaded': self.processor is not None
        }

def main():
    parser = argparse.ArgumentParser(description="Standardized Embedding Pipeline")
    parser.add_argument("--ai-dir", type=str, default=config.AI_IMAGES_DIR, help="Directory containing AI images")
    parser.add_argument("--real-dir", type=str, default=config.REAL_IMAGES_DIR, help="Directory containing real images")
    parser.add_argument("--output-dir", type=str, default=config.EMBEDDINGS_DIR, help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--version", type=str, default="1.0", help="Pipeline version")
    parser.add_argument("--model", type=str, default=config.CLIP_MODEL_NAME, help="CLIP model name")
    parser.add_argument("--validate", action="store_true", help="Validate existing embeddings")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(args.model, args.version)
    
    if args.validate:
        # Validate existing embeddings
        latest_path = Path(args.output_dir) / "latest_embeddings.npy"
        if latest_path.exists():
            embeddings, metadata = pipeline.load_embeddings_with_version(str(latest_path))
            if pipeline.validate_embeddings(embeddings):
                print("✅ Embeddings validation passed")
                print(f"📊 Embeddings info: {embeddings.shape}")
                print(f"📋 Metadata: {metadata.get('version', 'unknown')}")
            else:
                print("❌ Embeddings validation failed")
        else:
            print("❌ No embeddings found to validate")
        return
    
    # Process dataset
    ai_embeddings, real_embeddings, labels, processing_metadata = pipeline.process_dataset(args.ai_dir, args.real_dir)
    
    # Validate embeddings
    all_embeddings = np.vstack([ai_embeddings, real_embeddings]) if len(ai_embeddings) > 0 and len(real_embeddings) > 0 else np.array([])
    if not pipeline.validate_embeddings(all_embeddings):
        logger.error("Embedding validation failed")
        return
    
    # Save embeddings with version tracking
    output_path, metadata_path = pipeline.save_embeddings_with_version(
        ai_embeddings, real_embeddings, labels, processing_metadata, args.output_dir
    )
    
    print(f"✅ Successfully generated embeddings for {len(ai_embeddings)} AI images and {len(real_embeddings)} real images")
    print(f"📁 Embeddings saved to {output_path}")
    print(f"📋 Metadata saved to {metadata_path}")
    print(f"🔢 Pipeline version: {args.version}")

if __name__ == "__main__":
    main()
