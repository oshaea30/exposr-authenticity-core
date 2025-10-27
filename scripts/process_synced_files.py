"""
Process synced files from Google Drive
Automatically triggers embedding for images
"""
import sys
sys.path.insert(0, '..')

import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYNC_DIR = 'training_data/google_drive_sync/'

def main():
    """Process newly synced files"""
    sync_dir = Path(SYNC_DIR)
    images_dir = sync_dir / 'images'
    
    if not images_dir.exists():
        logger.warning("No images directory found")
        return
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        logger.info("No new images to process")
        return
    
    logger.info(f"📸 Found {len(image_files)} images to embed...")
    
    # Trigger embedding pipeline
    try:
        # Run embed_images.py with the synced images
        result = subprocess.run(
            ['python', '../embed_images.py', '--input', str(images_dir), '--output', 'embeddings/drive_synced/'],
            capture_output=True,
            text=True,
            cwd='..'
        )
        
        if result.returncode == 0:
            logger.info("✅ Images embedded successfully")
            print(result.stdout)
        else:
            logger.error("❌ Embedding failed")
            print(result.stderr)
            
    except Exception as e:
        logger.error(f"Failed to process images: {e}")

if __name__ == "__main__":
    main()

