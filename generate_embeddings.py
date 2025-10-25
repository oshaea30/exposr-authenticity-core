"""
CLIP Embedding Pipeline for Image Vectorization
"""
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import pickle
import argparse
from transformers import CLIPProcessor, CLIPModel
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class CLIPEmbeddingGenerator:
    def __init__(self, model_name: str = config.CLIP_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("CLIP model loaded successfully")
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess image for CLIP model"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use CLIP processor for consistent preprocessing
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate CLIP embedding for a single image"""
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
        """Generate embeddings for a batch of images"""
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
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
        
        return embeddings
    
    def process_dataset(self, ai_dir: str, real_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process entire dataset and generate embeddings"""
        logger.info("Starting dataset processing")
        
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
        
        return ai_array, real_array, labels
    
    def get_image_paths(self, directory: str) -> List[str]:
        """Get all image file paths from directory"""
        directory = Path(directory)
        image_paths = []
        
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        return [str(path) for path in image_paths]
    
    def save_embeddings(self, ai_embeddings: np.ndarray, real_embeddings: np.ndarray, 
                       labels: List[str], output_path: str):
        """Save embeddings and metadata"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine embeddings
        all_embeddings = np.vstack([ai_embeddings, real_embeddings]) if len(ai_embeddings) > 0 and len(real_embeddings) > 0 else np.array([])
        
        # Save embeddings
        np.save(output_path, all_embeddings)
        
        # Save metadata
        metadata = {
            'ai_count': len(ai_embeddings),
            'real_count': len(real_embeddings),
            'total_count': len(all_embeddings),
            'embedding_dimension': all_embeddings.shape[1] if len(all_embeddings) > 0 else 0,
            'labels': labels,
            'model_name': self.model_name
        }
        
        metadata_path = output_path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_embeddings(self, embeddings_path: str) -> Tuple[np.ndarray, dict]:
        """Load embeddings and metadata"""
        embeddings_path = Path(embeddings_path)
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        metadata_path = embeddings_path.with_suffix('.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return embeddings, metadata

def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for images")
    parser.add_argument("--ai-dir", type=str, default=config.AI_IMAGES_DIR, help="Directory containing AI images")
    parser.add_argument("--real-dir", type=str, default=config.REAL_IMAGES_DIR, help="Directory containing real images")
    parser.add_argument("--output", type=str, default=os.path.join(config.EMBEDDINGS_DIR, "image_vectors.npy"), help="Output path for embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize CLIP generator
    generator = CLIPEmbeddingGenerator()
    
    # Process dataset
    ai_embeddings, real_embeddings, labels = generator.process_dataset(args.ai_dir, args.real_dir)
    
    # Save embeddings
    generator.save_embeddings(ai_embeddings, real_embeddings, labels, args.output)
    
    print(f"Successfully generated embeddings for {len(ai_embeddings)} AI images and {len(real_embeddings)} real images")
    print(f"Embeddings saved to {args.output}")

if __name__ == "__main__":
    main()
