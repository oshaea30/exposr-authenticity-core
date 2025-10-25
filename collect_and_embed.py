"""
Master Collection and Embedding Script for Exposr
Runs both scrapers and immediately processes images through embedding pipeline
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json
import pandas as pd
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollectionAndEmbeddingPipeline:
    def __init__(self):
        self.start_time = datetime.now()
        self.collection_log = []
        self.embedding_log = []
        
        # Ensure directories exist
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            "data/ai",
            "data/real", 
            "embeddings",
            "models",
            "training_reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Log pipeline events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'data': data or {}
        }
        
        if event_type.startswith('collection'):
            self.collection_log.append(event)
        else:
            self.embedding_log.append(event)
        
        logger.info(f"{event_type}: {message}")
    
    def run_real_image_collection(self, count: int = 1000) -> bool:
        """Run real image collection"""
        logger.info("🖼️  Starting real image collection...")
        
        try:
            cmd = [
                sys.executable, 
                "scrape_public_real_images.py", 
                "--count", str(count)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                self.log_event("collection_real", f"Successfully collected real images", {
                    'count': count,
                    'stdout': result.stdout
                })
                return True
            else:
                self.log_event("collection_real", f"Real image collection failed", {
                    'error': result.stderr,
                    'returncode': result.returncode
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.log_event("collection_real", "Real image collection timed out", {})
            return False
        except Exception as e:
            self.log_event("collection_real", f"Real image collection error: {str(e)}", {})
            return False
    
    def run_ai_image_collection(self, count: int = 1000) -> bool:
        """Run AI image collection"""
        logger.info("🤖 Starting AI image collection...")
        
        try:
            cmd = [
                sys.executable, 
                "scrape_public_ai_images.py", 
                "--count", str(count)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                self.log_event("collection_ai", f"Successfully collected AI images", {
                    'count': count,
                    'stdout': result.stdout
                })
                return True
            else:
                self.log_event("collection_ai", f"AI image collection failed", {
                    'error': result.stderr,
                    'returncode': result.returncode
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.log_event("collection_ai", "AI image collection timed out", {})
            return False
        except Exception as e:
            self.log_event("collection_ai", f"AI image collection error: {str(e)}", {})
            return False
    
    def run_embedding_generation(self, version: str = None) -> bool:
        """Run embedding generation"""
        logger.info("🧠 Starting embedding generation...")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            cmd = [
                sys.executable, 
                "embed_images.py", 
                "--version", version,
                "--ai-dir", "data/ai",
                "--real-dir", "data/real"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                self.log_event("embedding_generation", f"Successfully generated embeddings", {
                    'version': version,
                    'stdout': result.stdout
                })
                return True
            else:
                self.log_event("embedding_generation", f"Embedding generation failed", {
                    'error': result.stderr,
                    'returncode': result.returncode
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.log_event("embedding_generation", "Embedding generation timed out", {})
            return False
        except Exception as e:
            self.log_event("embedding_generation", f"Embedding generation error: {str(e)}", {})
            return False
    
    def update_dataset_manifest(self):
        """Update dataset manifest with collection statistics"""
        manifest_file = Path("dataset_manifest.json")
        
        # Get current statistics
        ai_count = len(list(Path("data/ai").glob("*.jpg"))) + len(list(Path("data/ai").glob("*.png")))
        real_count = len(list(Path("data/real").glob("*.jpg"))) + len(list(Path("data/real").glob("*.png")))
        
        # Read metadata if available
        metadata_file = Path("dataset_log.csv")
        metadata_count = 0
        if metadata_file.exists():
            try:
                df = pd.read_csv(metadata_file)
                metadata_count = len(df)
            except Exception as e:
                logger.warning(f"Could not read metadata file: {e}")
        
        manifest = {
            'created_at': datetime.now().isoformat(),
            'total_images': ai_count + real_count,
            'ai_images': ai_count,
            'real_images': real_count,
            'metadata_entries': metadata_count,
            'collection_log': self.collection_log,
            'embedding_log': self.embedding_log,
            'pipeline_version': '1.0'
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📋 Dataset manifest updated: {ai_count} AI, {real_count} real images")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get collection summary statistics"""
        ai_count = len(list(Path("data/ai").glob("*.jpg"))) + len(list(Path("data/ai").glob("*.png")))
        real_count = len(list(Path("data/real").glob("*.jpg"))) + len(list(Path("data/real").glob("*.png")))
        
        # Check if embeddings exist
        embeddings_file = Path("embeddings/latest_embeddings.npy")
        embeddings_exist = embeddings_file.exists()
        
        # Check metadata
        metadata_file = Path("dataset_log.csv")
        metadata_count = 0
        if metadata_file.exists():
            try:
                df = pd.read_csv(metadata_file)
                metadata_count = len(df)
            except Exception:
                pass
        
        return {
            'ai_images': ai_count,
            'real_images': real_count,
            'total_images': ai_count + real_count,
            'embeddings_generated': embeddings_exist,
            'metadata_entries': metadata_count,
            'collection_duration': str(datetime.now() - self.start_time),
            'pipeline_status': 'complete' if embeddings_exist else 'incomplete'
        }
    
    def run_complete_pipeline(self, ai_count: int = 1000, real_count: int = 1000, version: str = None):
        """Run the complete collection and embedding pipeline"""
        logger.info("🚀 Starting complete collection and embedding pipeline")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Collect real images
        logger.info(f"📥 Step 1: Collecting {real_count} real images...")
        real_success = self.run_real_image_collection(real_count)
        
        if not real_success:
            logger.error("❌ Real image collection failed")
            return False
        
        # Step 2: Collect AI images
        logger.info(f"🤖 Step 2: Collecting {ai_count} AI images...")
        ai_success = self.run_ai_image_collection(ai_count)
        
        if not ai_success:
            logger.error("❌ AI image collection failed")
            return False
        
        # Step 3: Generate embeddings
        logger.info("🧠 Step 3: Generating embeddings...")
        embedding_success = self.run_embedding_generation(version)
        
        if not embedding_success:
            logger.error("❌ Embedding generation failed")
            return False
        
        # Step 4: Update manifest
        logger.info("📋 Step 4: Updating dataset manifest...")
        self.update_dataset_manifest()
        
        # Step 5: Generate summary
        summary = self.get_collection_summary()
        
        logger.info("✅ Complete pipeline finished successfully!")
        logger.info(f"📊 Final Summary:")
        logger.info(f"  • AI Images: {summary['ai_images']}")
        logger.info(f"  • Real Images: {summary['real_images']}")
        logger.info(f"  • Total Images: {summary['total_images']}")
        logger.info(f"  • Embeddings Generated: {summary['embeddings_generated']}")
        logger.info(f"  • Metadata Entries: {summary['metadata_entries']}")
        logger.info(f"  • Duration: {summary['collection_duration']}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Master collection and embedding pipeline")
    parser.add_argument("--ai-count", type=int, default=1000, help="Number of AI images to collect")
    parser.add_argument("--real-count", type=int, default=1000, help="Number of real images to collect")
    parser.add_argument("--version", type=str, help="Pipeline version (default: timestamp)")
    parser.add_argument("--real-only", action="store_true", help="Only collect real images")
    parser.add_argument("--ai-only", action="store_true", help="Only collect AI images")
    parser.add_argument("--embed-only", action="store_true", help="Only generate embeddings (skip collection)")
    parser.add_argument("--summary", action="store_true", help="Show collection summary")
    
    args = parser.parse_args()
    
    pipeline = CollectionAndEmbeddingPipeline()
    
    if args.summary:
        summary = pipeline.get_collection_summary()
        print(f"\n📊 Collection Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return
    
    if args.embed_only:
        # Only generate embeddings
        success = pipeline.run_embedding_generation(args.version)
        if success:
            pipeline.update_dataset_manifest()
            print("✅ Embedding generation completed")
        else:
            print("❌ Embedding generation failed")
        return
    
    if args.real_only:
        # Only collect real images
        success = pipeline.run_real_image_collection(args.real_count)
        if success:
            pipeline.run_embedding_generation(args.version)
            pipeline.update_dataset_manifest()
            print("✅ Real image collection completed")
        else:
            print("❌ Real image collection failed")
        return
    
    if args.ai_only:
        # Only collect AI images
        success = pipeline.run_ai_image_collection(args.ai_count)
        if success:
            pipeline.run_embedding_generation(args.version)
            pipeline.update_dataset_manifest()
            print("✅ AI image collection completed")
        else:
            print("❌ AI image collection failed")
        return
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline(args.ai_count, args.real_count, args.version)
    
    if success:
        print("\n🎉 Complete pipeline finished successfully!")
        print("📁 Check the following files:")
        print("  • data/ai/ - AI-generated images")
        print("  • data/real/ - Real images")
        print("  • embeddings/latest_embeddings.npy - Generated embeddings")
        print("  • dataset_log.csv - Image metadata")
        print("  • dataset_manifest.json - Pipeline summary")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()
