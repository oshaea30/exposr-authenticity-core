"""
Public Domain Real Image Scraper for Exposr
Scrapes only from public domain and license-free sources
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

class PublicRealImageScraper:
    def __init__(self, output_dir: str = "data/real"):
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
                'real',
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
    
    def get_image_hash(self, image_path: str) -> str:
        """Get hash of image content to detect duplicates"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
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
    
    def scrape_wikimedia_commons(self, count: int = 100) -> int:
        """Scrape public domain images from Wikimedia Commons"""
        logger.info(f"Scraping Wikimedia Commons for {count} public domain images")
        
        collected = 0
        base_url = "https://commons.wikimedia.org/w/api.php"
        
        # Categories with public domain images
        categories = [
            "Category:Public_domain_images",
            "Category:CC0_images", 
            "Category:Public_domain_photographs",
            "Category:Public_domain_artwork"
        ]
        
        for category in categories:
            if collected >= count:
                break
                
            logger.info(f"Searching category: {category}")
            
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmtype': 'file',
                'cmnamespace': 6,  # File namespace
                'cmlimit': 50,
                'cmprop': 'title'
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data and 'categorymembers' in data['query']:
                    for member in data['query']['categorymembers']:
                        if collected >= count:
                            break
                            
                        title = member['title']
                        if self.is_image_file(title):
                            # Get file info
                            file_url = self.get_wikimedia_file_url(title)
                            if file_url:
                                filename = f"wikimedia_{collected:06d}.jpg"
                                if self.download_image(file_url, filename):
                                    file_size = os.path.getsize(self.output_dir / filename)
                                    self.log_image_metadata(filename, "wikimedia_commons", "public_domain", file_url, file_size)
                                    collected += 1
                                    logger.info(f"Downloaded Wikimedia image {collected}/{count}")
                                
                                self.rate_limit()
                
            except Exception as e:
                logger.error(f"Error scraping category {category}: {e}")
                continue
        
        logger.info(f"Collected {collected} images from Wikimedia Commons")
        return collected
    
    def get_wikimedia_file_url(self, title: str) -> Optional[str]:
        """Get direct URL for a Wikimedia file"""
        base_url = "https://commons.wikimedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'imageinfo',
            'iiprop': 'url'
        }
        
        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'imageinfo' in page_data:
                    imageinfo = page_data['imageinfo'][0]
                    return imageinfo.get('url')
                    
        except Exception as e:
            logger.debug(f"Error getting file URL for {title}: {e}")
            
        return None
    
    def scrape_rawpixel_public_domain(self, count: int = 100) -> int:
        """Scrape public domain images from RawPixel"""
        logger.info(f"Scraping RawPixel public domain for {count} images")
        
        collected = 0
        base_url = "https://www.rawpixel.com/free-images"
        
        try:
            # Get the public domain page
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find image containers
            image_containers = soup.find_all('div', class_='image-container')[:count]
            
            for container in image_containers:
                if collected >= count:
                    break
                    
                try:
                    # Find image URL
                    img_tag = container.find('img')
                    if img_tag:
                        img_url = img_tag.get('src') or img_tag.get('data-src')
                        if img_url:
                            # Convert to high-res URL if possible
                            if 'thumb' in img_url:
                                img_url = img_url.replace('thumb/', '').replace('/thumb', '')
                            
                            filename = f"rawpixel_{collected:06d}.jpg"
                            if self.download_image(img_url, filename):
                                file_size = os.path.getsize(self.output_dir / filename)
                                self.log_image_metadata(filename, "rawpixel", "public_domain", img_url, file_size)
                                collected += 1
                                logger.info(f"Downloaded RawPixel image {collected}/{count}")
                            
                            self.rate_limit()
                            
                except Exception as e:
                    logger.debug(f"Error processing RawPixel image: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping RawPixel: {e}")
        
        logger.info(f"Collected {collected} images from RawPixel")
        return collected
    
    def scrape_librestock(self, count: int = 100) -> int:
        """Scrape free images from Librestock"""
        logger.info(f"Scraping Librestock for {count} images")
        
        collected = 0
        base_url = "https://librestock.com"
        
        try:
            # Get the main page
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find image links
            image_links = soup.find_all('a', href=True)[:count]
            
            for link in image_links:
                if collected >= count:
                    break
                    
                try:
                    href = link.get('href')
                    if href and ('jpg' in href.lower() or 'png' in href.lower()):
                        # Get full URL
                        img_url = urljoin(base_url, href)
                        
                        filename = f"librestock_{collected:06d}.jpg"
                        if self.download_image(img_url, filename):
                            file_size = os.path.getsize(self.output_dir / filename)
                            self.log_image_metadata(filename, "librestock", "free", img_url, file_size)
                            collected += 1
                            logger.info(f"Downloaded Librestock image {collected}/{count}")
                        
                        self.rate_limit()
                        
                except Exception as e:
                    logger.debug(f"Error processing Librestock image: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Librestock: {e}")
        
        logger.info(f"Collected {collected} images from Librestock")
        return collected
    
    def is_image_file(self, title: str) -> bool:
        """Check if Wikimedia title is an image file"""
        title_lower = title.lower()
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            if title_lower.endswith(ext):
                return True
        return False
    
    def scrape_all_sources(self, target_count: int = 1000) -> int:
        """Scrape from all public domain sources"""
        logger.info(f"Starting public domain real image collection for {target_count} images")
        
        total_collected = 0
        
        # Distribute target across sources
        per_source = target_count // 3
        
        # Wikimedia Commons
        wikimedia_count = self.scrape_wikimedia_commons(per_source)
        total_collected += wikimedia_count
        
        # RawPixel
        rawpixel_count = self.scrape_rawpixel_public_domain(per_source)
        total_collected += rawpixel_count
        
        # Librestock
        librestock_count = self.scrape_librestock(per_source)
        total_collected += librestock_count
        
        logger.info(f"Total collected: {total_collected} public domain real images")
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
    parser = argparse.ArgumentParser(description="Scrape public domain real images")
    parser.add_argument("--count", type=int, default=1000, help="Target number of images to collect")
    parser.add_argument("--output", type=str, default="data/real", help="Output directory")
    parser.add_argument("--wikimedia-only", action="store_true", help="Only scrape Wikimedia Commons")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    scraper = PublicRealImageScraper(args.output)
    
    if args.stats:
        stats = scraper.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    if args.wikimedia_only:
        collected = scraper.scrape_wikimedia_commons(args.count)
    else:
        collected = scraper.scrape_all_sources(args.count)
    
    print(f"✅ Successfully collected {collected} public domain real images")
    print(f"📁 Images saved to {args.output}")
    print(f"📋 Metadata logged to dataset_log.csv")

if __name__ == "__main__":
    main()
