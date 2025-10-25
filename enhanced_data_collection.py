"""
Enhanced Data Collection Pipeline
"""
import os
import requests
import time
import random
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
import argparse
from typing import List, Optional, Dict, Any
import json
import hashlib
from PIL import Image
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    def __init__(self, output_dir: str, data_type: str):
        self.output_dir = Path(output_dir)
        self.data_type = data_type  # 'ai' or 'real'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.metadata_file = self.output_dir / f"{data_type}_metadata.json"
        self.load_metadata()
        
    def load_metadata(self):
        """Load existing metadata to avoid duplicates"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'collected_images': [],
                'failed_downloads': [],
                'total_count': 0,
                'last_updated': None
            }
    
    def save_metadata(self):
        """Save metadata to file"""
        self.metadata['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def validate_image(self, image_path: str) -> bool:
        """Validate downloaded image quality"""
        try:
            with Image.open(image_path) as img:
                # Check image dimensions
                width, height = img.size
                if width < 64 or height < 64:
                    logger.warning(f"Image too small: {width}x{height}")
                    return False
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    logger.warning(f"Extreme aspect ratio: {aspect_ratio}")
                    return False
                
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size < 1024:  # Less than 1KB
                    logger.warning(f"File too small: {file_size} bytes")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def get_image_hash(self, image_path: str) -> str:
        """Get hash of image content to detect duplicates"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def scrape_ai_sources(self, target_count: int = 1000) -> int:
        """Scrape from multiple AI image sources"""
        logger.info(f"Starting AI image collection for {target_count} images")
        
        sources = [
            self.scrape_lexica_art,
            self.scrape_artbreeder,
            self.scrape_midjourney_gallery,
            self.scrape_dalle_images,
            self.scrape_stable_diffusion_images
        ]
        
        collected = 0
        for source_func in sources:
            if collected >= target_count:
                break
            
            try:
                remaining = target_count - collected
                count = source_func(min(remaining, 200))  # Limit per source
                collected += count
                logger.info(f"Collected {count} images from {source_func.__name__}")
                
                # Rate limiting between sources
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Error with {source_func.__name__}: {e}")
                continue
        
        logger.info(f"Total AI images collected: {collected}")
        return collected
    
    def scrape_real_sources(self, target_count: int = 1000) -> int:
        """Scrape from multiple real image sources"""
        logger.info(f"Starting real image collection for {target_count} images")
        
        sources = [
            self.scrape_wikimedia_commons,
            self.scrape_unsplash,
            self.scrape_pexels,
            self.scrape_flickr,
            self.scrape_public_domains
        ]
        
        collected = 0
        for source_func in sources:
            if collected >= target_count:
                break
            
            try:
                remaining = target_count - collected
                count = source_func(min(remaining, 200))  # Limit per source
                collected += count
                logger.info(f"Collected {count} images from {source_func.__name__}")
                
                # Rate limiting between sources
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Error with {source_func.__name__}: {e}")
                continue
        
        logger.info(f"Total real images collected: {collected}")
        return collected
    
    def scrape_lexica_art(self, count: int) -> int:
        """Enhanced Lexica.art scraper"""
        logger.info(f"Scraping Lexica.art for {count} images")
        
        # Use Lexica API if available, otherwise fallback to web scraping
        try:
            return self.scrape_lexica_api(count)
        except Exception:
            logger.warning("Lexica API failed, using web scraping")
            return self.scrape_lexica_web(count)
    
    def scrape_lexica_api(self, count: int) -> int:
        """Scrape using Lexica API"""
        # Lexica API endpoint (if available)
        api_url = "https://lexica.art/api/v1/search"
        
        collected = 0
        page = 1
        
        while collected < count:
            params = {
                'q': 'portrait landscape architecture nature',  # Diverse prompts
                'page': page,
                'per_page': min(50, count - collected)
            }
            
            try:
                response = self.session.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for image_data in data.get('images', []):
                    if collected >= count:
                        break
                    
                    image_url = image_data.get('src')
                    if image_url and self.download_and_validate(image_url):
                        collected += 1
                
                page += 1
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Lexica API error: {e}")
                break
        
        return collected
    
    def scrape_lexica_web(self, count: int) -> int:
        """Fallback web scraping for Lexica"""
        # Implementation similar to existing scraper but enhanced
        collected = 0
        # ... (implement enhanced web scraping)
        return collected
    
    def scrape_artbreeder(self, count: int) -> int:
        """Enhanced Artbreeder scraper"""
        logger.info(f"Scraping Artbreeder for {count} images")
        
        collected = 0
        # Enhanced implementation with better error handling
        # ... (implement enhanced Artbreeder scraping)
        return collected
    
    def scrape_midjourney_gallery(self, count: int) -> int:
        """Enhanced Midjourney gallery scraper"""
        logger.info(f"Scraping Midjourney gallery for {count} images")
        
        collected = 0
        # Enhanced implementation
        # ... (implement enhanced Midjourney scraping)
        return collected
    
    def scrape_dalle_images(self, count: int) -> int:
        """Scrape DALL-E generated images"""
        logger.info(f"Scraping DALL-E images for {count} images")
        
        collected = 0
        # Implementation for DALL-E image sources
        # ... (implement DALL-E scraping)
        return collected
    
    def scrape_stable_diffusion_images(self, count: int) -> int:
        """Scrape Stable Diffusion generated images"""
        logger.info(f"Scraping Stable Diffusion images for {count} images")
        
        collected = 0
        # Implementation for Stable Diffusion sources
        # ... (implement Stable Diffusion scraping)
        return collected
    
    def scrape_wikimedia_commons(self, count: int) -> int:
        """Enhanced Wikimedia Commons scraper"""
        logger.info(f"Scraping Wikimedia Commons for {count} images")
        
        collected = 0
        # Enhanced implementation with better API usage
        # ... (implement enhanced Wikimedia scraping)
        return collected
    
    def scrape_unsplash(self, count: int) -> int:
        """Enhanced Unsplash scraper"""
        logger.info(f"Scraping Unsplash for {count} images")
        
        collected = 0
        # Enhanced implementation with API key usage
        # ... (implement enhanced Unsplash scraping)
        return collected
    
    def scrape_pexels(self, count: int) -> int:
        """Enhanced Pexels scraper"""
        logger.info(f"Scraping Pexels for {count} images")
        
        collected = 0
        # Enhanced implementation
        # ... (implement enhanced Pexels scraping)
        return collected
    
    def scrape_flickr(self, count: int) -> int:
        """Scrape Flickr public images"""
        logger.info(f"Scraping Flickr for {count} images")
        
        collected = 0
        # Implementation for Flickr scraping
        # ... (implement Flickr scraping)
        return collected
    
    def scrape_public_domains(self, count: int) -> int:
        """Scrape from public domain image sources"""
        logger.info(f"Scraping public domain sources for {count} images")
        
        collected = 0
        # Implementation for public domain sources
        # ... (implement public domain scraping)
        return collected
    
    def download_and_validate(self, url: str) -> bool:
        """Download image and validate quality"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Generate filename
            filename = f"{self.data_type}_{len(self.metadata['collected_images']):06d}.jpg"
            filepath = self.output_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Validate image
            if not self.validate_image(str(filepath)):
                filepath.unlink()  # Delete invalid image
                return False
            
            # Check for duplicates
            image_hash = self.get_image_hash(str(filepath))
            if image_hash in [img['hash'] for img in self.metadata['collected_images']]:
                filepath.unlink()  # Delete duplicate
                return False
            
            # Add to metadata
            self.metadata['collected_images'].append({
                'filename': filename,
                'url': url,
                'hash': image_hash,
                'size': os.path.getsize(filepath),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            self.metadata['total_count'] += 1
            self.save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            self.metadata['failed_downloads'].append({
                'url': url,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            return False
    
    def clean_dataset(self):
        """Clean and validate existing dataset"""
        logger.info("Cleaning existing dataset...")
        
        cleaned_count = 0
        for image_file in self.output_dir.glob("*.jpg"):
            if not self.validate_image(str(image_file)):
                logger.warning(f"Removing invalid image: {image_file}")
                image_file.unlink()
                cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} invalid images")
        return cleaned_count
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        total_files = len(list(self.output_dir.glob("*.jpg")))
        
        stats = {
            'total_images': total_files,
            'metadata_entries': len(self.metadata['collected_images']),
            'failed_downloads': len(self.metadata['failed_downloads']),
            'last_updated': self.metadata['last_updated'],
            'data_type': self.data_type
        }
        
        # Calculate size statistics
        total_size = sum(f.stat().st_size for f in self.output_dir.glob("*.jpg"))
        stats['total_size_mb'] = total_size / (1024 * 1024)
        stats['avg_size_mb'] = stats['total_size_mb'] / total_files if total_files > 0 else 0
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Enhanced Data Collection Pipeline")
    parser.add_argument("--type", choices=["ai", "real"], required=True, help="Type of images to collect")
    parser.add_argument("--count", type=int, default=1000, help="Target number of images")
    parser.add_argument("--clean", action="store_true", help="Clean existing dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = config.AI_IMAGES_DIR if args.type == "ai" else config.REAL_IMAGES_DIR
    
    collector = EnhancedDataCollector(output_dir, args.type)
    
    if args.stats:
        stats = collector.get_dataset_stats()
        print(f"\n📊 Dataset Statistics for {args.type.upper()} images:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    if args.clean:
        cleaned = collector.clean_dataset()
        print(f"Cleaned {cleaned} invalid images")
        return
    
    # Collect images
    if args.type == "ai":
        collected = collector.scrape_ai_sources(args.count)
    else:
        collected = collector.scrape_real_sources(args.count)
    
    print(f"✅ Collected {collected} {args.type} images")
    
    # Show final stats
    stats = collector.get_dataset_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
