"""
Public AI Image Scraper for Exposr
Scrapes AI-generated images from public sources
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
import csv
from datetime import datetime
from PIL import Image
import hashlib
from bs4 import BeautifulSoup
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class PublicAIImageScraper:
    def __init__(self, output_dir: str = "data/ai"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Metadata logging
        self.metadata_file = Path("dataset_log.csv")
        self.init_metadata_file()
        
        # Rate limiting
        self.min_delay = 1.0
        self.max_delay = 2.0
        
    def init_metadata_file(self):
        """Initialize metadata CSV file if it doesn't exist"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'source', 'label', 'license', 'timestamp', 'url', 'file_size'])
    
    def log_image_metadata(self, filename: str, source: str, license: str, url: str, file_size: int):
        """Log image metadata to CSV"""
        with open(self.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,
                source,
                'ai',
                license,
                datetime.now().isoformat(),
                url,
                file_size
            ])
    
    def rate_limit(self):
        """Apply rate limiting"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
    
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
                
                return True
                
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download image from URL with validation"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/webp']):
                logger.warning(f"Invalid content type for {url}: {content_type}")
                return False
            
            # Check file size
            if len(response.content) > config.MAX_IMAGE_SIZE:
                logger.warning(f"Image too large: {url}")
                return False
                
            # Save image
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Validate image
            if not self.validate_image(str(filepath)):
                filepath.unlink()  # Delete invalid image
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def scrape_lexica_art(self, count: int = 100) -> int:
        """Scrape AI images from Lexica.art"""
        logger.info(f"Scraping Lexica.art for {count} AI images")
        
        collected = 0
        base_url = "https://lexica.art"
        
        try:
            # Get the main page
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find image containers - Lexica uses specific classes
            image_containers = soup.find_all('div', class_='image-container')[:count]
            
            # If no specific containers found, look for img tags
            if not image_containers:
                image_containers = soup.find_all('img')[:count]
            
            for container in image_containers:
                if collected >= count:
                    break
                    
                try:
                    # Extract image URL
                    if container.name == 'img':
                        img_url = container.get('src') or container.get('data-src')
                    else:
                        img_tag = container.find('img')
                        img_url = img_tag.get('src') or img_tag.get('data-src') if img_tag else None
                    
                    if img_url:
                        # Convert to full URL if needed
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif img_url.startswith('/'):
                            img_url = urljoin(base_url, img_url)
                        elif not img_url.startswith('http'):
                            img_url = urljoin(base_url, img_url)
                        
                        filename = f"lexica_{collected:06d}.jpg"
                        if self.download_image(img_url, filename):
                            file_size = os.path.getsize(self.output_dir / filename)
                            self.log_image_metadata(filename, "lexica_art", "public", img_url, file_size)
                            collected += 1
                            logger.info(f"Downloaded Lexica image {collected}/{count}")
                        
                        self.rate_limit()
                        
                except Exception as e:
                    logger.debug(f"Error processing Lexica image: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Lexica.art: {e}")
        
        logger.info(f"Collected {collected} images from Lexica.art")
        return collected
    
    def scrape_artbreeder(self, count: int = 100) -> int:
        """Scrape AI images from Artbreeder (public gallery)"""
        logger.info(f"Scraping Artbreeder for {count} AI images")
        
        collected = 0
        base_url = "https://www.artbreeder.com/browse"
        
        try:
            # Get the browse page
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find image containers
            image_containers = soup.find_all('div', class_='image-container')[:count]
            
            # If no specific containers, look for img tags
            if not image_containers:
                image_containers = soup.find_all('img')[:count]
            
            for container in image_containers:
                if collected >= count:
                    break
                    
                try:
                    # Extract image URL
                    if container.name == 'img':
                        img_url = container.get('src') or container.get('data-src')
                    else:
                        img_tag = container.find('img')
                        img_url = img_tag.get('src') or img_tag.get('data-src') if img_tag else None
                    
                    if img_url:
                        # Convert to full URL if needed
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif img_url.startswith('/'):
                            img_url = urljoin(base_url, img_url)
                        elif not img_url.startswith('http'):
                            img_url = urljoin(base_url, img_url)
                        
                        filename = f"artbreeder_{collected:06d}.jpg"
                        if self.download_image(img_url, filename):
                            file_size = os.path.getsize(self.output_dir / filename)
                            self.log_image_metadata(filename, "artbreeder", "public", img_url, file_size)
                            collected += 1
                            logger.info(f"Downloaded Artbreeder image {collected}/{count}")
                        
                        self.rate_limit()
                        
                except Exception as e:
                    logger.debug(f"Error processing Artbreeder image: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Artbreeder: {e}")
        
        logger.info(f"Collected {collected} images from Artbreeder")
        return collected
    
    def scrape_reddit_midjourney(self, count: int = 100) -> int:
        """Scrape Midjourney images from Reddit r/midjourney"""
        logger.info(f"Scraping Reddit r/midjourney for {count} AI images")
        
        collected = 0
        base_url = "https://www.reddit.com/r/midjourney.json"
        
        try:
            # Get Reddit JSON feed
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            posts = data.get('data', {}).get('children', [])
            
            for post in posts:
                if collected >= count:
                    break
                    
                try:
                    post_data = post.get('data', {})
                    
                    # Check if post has an image
                    if post_data.get('post_hint') == 'image':
                        img_url = post_data.get('url')
                        
                        # Skip Reddit's own image hosting
                        if img_url and not img_url.startswith('https://preview.redd.it/'):
                            filename = f"reddit_midjourney_{collected:06d}.jpg"
                            if self.download_image(img_url, filename):
                                file_size = os.path.getsize(self.output_dir / filename)
                                self.log_image_metadata(filename, "reddit_midjourney", "public", img_url, file_size)
                                collected += 1
                                logger.info(f"Downloaded Reddit Midjourney image {collected}/{count}")
                            
                            self.rate_limit()
                    
                    # Also check for gallery posts
                    elif post_data.get('is_gallery'):
                        gallery_data = post_data.get('gallery_data', {}).get('items', [])
                        for item in gallery_data[:3]:  # Limit to 3 images per gallery
                            if collected >= count:
                                break
                                
                            media_id = item.get('media_id')
                            if media_id:
                                img_url = f"https://i.redd.it/{media_id}.jpg"
                                filename = f"reddit_midjourney_{collected:06d}.jpg"
                                if self.download_image(img_url, filename):
                                    file_size = os.path.getsize(self.output_dir / filename)
                                    self.log_image_metadata(filename, "reddit_midjourney", "public", img_url, file_size)
                                    collected += 1
                                    logger.info(f"Downloaded Reddit Midjourney gallery image {collected}/{count}")
                                
                                self.rate_limit()
                        
                except Exception as e:
                    logger.debug(f"Error processing Reddit post: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
        
        logger.info(f"Collected {collected} images from Reddit r/midjourney")
        return collected
    
    def scrape_all_sources(self, target_count: int = 1000) -> int:
        """Scrape from all AI image sources"""
        logger.info(f"Starting AI image collection for {target_count} images")
        
        total_collected = 0
        
        # Distribute target across sources
        per_source = target_count // 3
        
        # Lexica.art
        lexica_count = self.scrape_lexica_art(per_source)
        total_collected += lexica_count
        
        # Artbreeder
        artbreeder_count = self.scrape_artbreeder(per_source)
        total_collected += artbreeder_count
        
        # Reddit r/midjourney
        reddit_count = self.scrape_reddit_midjourney(per_source)
        total_collected += reddit_count
        
        logger.info(f"Total collected: {total_collected} AI images")
        return total_collected
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        total_files = len(list(self.output_dir.glob("*.jpg"))) + len(list(self.output_dir.glob("*.png")))
        
        # Read metadata file
        metadata_count = 0
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                reader = csv.reader(f)
                metadata_count = sum(1 for row in reader) - 1  # Subtract header
        
        return {
            'total_images': total_files,
            'metadata_entries': metadata_count,
            'output_directory': str(self.output_dir),
            'metadata_file': str(self.metadata_file)
        }

def main():
    parser = argparse.ArgumentParser(description="Scrape AI-generated images from public sources")
    parser.add_argument("--count", type=int, default=1000, help="Target number of images to collect")
    parser.add_argument("--output", type=str, default="data/ai", help="Output directory")
    parser.add_argument("--lexica-only", action="store_true", help="Only scrape Lexica.art")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    scraper = PublicAIImageScraper(args.output)
    
    if args.stats:
        stats = scraper.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    if args.lexica_only:
        collected = scraper.scrape_lexica_art(args.count)
    else:
        collected = scraper.scrape_all_sources(args.count)
    
    print(f"✅ Successfully collected {collected} AI-generated images")
    print(f"📁 Images saved to {args.output}")
    print(f"📋 Metadata logged to dataset_log.csv")

if __name__ == "__main__":
    main()
