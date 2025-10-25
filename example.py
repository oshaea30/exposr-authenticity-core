"""
Example script demonstrating the complete Exposr Authenticity Core pipeline
"""
import os
import sys
import logging
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete ML pipeline from start to finish"""
    logger.info("🚀 Starting complete Exposr Authenticity Core pipeline...")
    
    # Step 1: Scrape AI images
    logger.info("📥 Step 1: Scraping AI images...")
    os.system("python scraper/scrape_ai.py --count 50")
    
    # Step 2: Scrape real images
    logger.info("📥 Step 2: Scraping real images...")
    os.system("python scraper/scrape_real.py --count 50")
    
    # Step 3: Generate embeddings
    logger.info("🧠 Step 3: Generating CLIP embeddings...")
    os.system("python generate_embeddings.py")
    
    # Step 4: Train classifier
    logger.info("🎯 Step 4: Training classifier...")
    os.system("python train.py")
    
    # Step 5: Test classification
    logger.info("🔍 Step 5: Testing classification...")
    
    # Find a test image
    ai_dir = Path(config.AI_IMAGES_DIR)
    real_dir = Path(config.REAL_IMAGES_DIR)
    
    test_images = []
    if ai_dir.exists():
        ai_images = list(ai_dir.glob("*"))
        if ai_images:
            test_images.append(str(ai_images[0]))
    
    if real_dir.exists():
        real_images = list(real_dir.glob("*"))
        if real_images:
            test_images.append(str(real_images[0]))
    
    if test_images:
        for img in test_images:
            logger.info(f"Testing image: {img}")
            os.system(f"python classify.py --image {img}")
    
    logger.info("✅ Pipeline completed successfully!")
    logger.info("🎉 Your AI/Real image classifier is ready!")
    
    # Show next steps
    logger.info("\n📋 Next steps:")
    logger.info("1. Run 'python app.py' to start the REST API")
    logger.info("2. Run 'python scheduler.py' to start automated retraining")
    logger.info("3. Use the API endpoints to classify new images")

def run_quick_demo():
    """Run a quick demo with minimal data"""
    logger.info("⚡ Running quick demo...")
    
    # Use smaller counts for demo
    logger.info("📥 Scraping demo data...")
    os.system("python scraper/scrape_ai.py --count 10")
    os.system("python scraper/scrape_real.py --count 10")
    
    logger.info("🧠 Generating embeddings...")
    os.system("python generate_embeddings.py")
    
    logger.info("🎯 Training classifier...")
    os.system("python train.py --no-optimize")  # Skip hyperparameter optimization for speed
    
    logger.info("🔍 Testing classification...")
    
    # Test with available images
    ai_dir = Path(config.AI_IMAGES_DIR)
    if ai_dir.exists():
        ai_images = list(ai_dir.glob("*"))
        if ai_images:
            logger.info(f"Testing AI image: {ai_images[0]}")
            os.system(f"python classify.py --image {ai_images[0]}")
    
    logger.info("✅ Quick demo completed!")

def show_usage():
    """Show usage information"""
    print("""
🎯 Exposr Authenticity Core - Usage Examples

1. Complete Pipeline (recommended for production):
   python example.py --complete

2. Quick Demo (for testing):
   python example.py --demo

3. Individual Steps:
   python scraper/scrape_ai.py --count 100
   python scraper/scrape_real.py --count 100
   python generate_embeddings.py
   python train.py
   python classify.py --image path/to/image.jpg

4. Start API Server:
   python app.py

5. Start Training Scheduler:
   python scheduler.py

6. Run Tests:
   python test.py

7. Setup (first time only):
   ./setup.sh
""")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1]
    
    if command == "--complete":
        run_complete_pipeline()
    elif command == "--demo":
        run_quick_demo()
    elif command == "--help":
        show_usage()
    else:
        print(f"Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main()
