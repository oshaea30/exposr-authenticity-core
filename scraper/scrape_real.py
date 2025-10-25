"""
Real Image Scraper for Wikimedia Commons and Unsplash
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
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class RealImageScraper:
    def __init__(self, output_dir: str = config.REAL_IMAGES_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def scrape_wikimedia_commons(self, count: int = 100) -> List[str]:
        """Scrape images from Wikimedia Commons using their API"""
        logger.info(f"Scraping Wikimedia Commons for {count} images")
        
        image_urls = []
        base_url = "https://commons.wikimedia.org/w/api.php"
        
        # Categories to search for diverse real images
        categories = [
            "Photographs by subject",
            "Nature photographs",
            "Portrait photographs",
            "Architecture photographs",
            "Street photography",
            "Landscape photographs",
            "Wildlife photographs"
        ]
        
        for category in categories:
            if len(image_urls) >= count:
                break
                
            logger.info(f"Searching category: {category}")
            
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmtype': 'file',
                'cmnamespace': 6,  # File namespace
                'cmlimit': 50,
                'cmprop': 'title'
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data and 'categorymembers' in data['query']:
                    for member in data['query']['categorymembers']:
                        if len(image_urls) >= count:
                            break
                            
                        title = member['title']
                        if self.is_image_file(title):
                            # Get file info
                            file_url = self.get_wikimedia_file_url(title)
                            if file_url:
                                image_urls.append(file_url)
                                
                time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
                
            except Exception as e:
                logger.error(f"Error scraping category {category}: {e}")
                continue
                
        logger.info(f"Found {len(image_urls)} Wikimedia images")
        return image_urls[:count]
    
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
            response = self.session.get(base_url, params=params, timeout=config.REQUEST_TIMEOUT)
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
    
    def scrape_unsplash(self, count: int = 100) -> List[str]:
        """Scrape images from Unsplash using their API"""
        logger.info(f"Scraping Unsplash for {count} images")
        
        image_urls = []
        base_url = "https://api.unsplash.com/photos/random"
        
        # Unsplash access key (you can get one for free)
        access_key = os.getenv('UNSPLASH_ACCESS_KEY')
        if not access_key:
            logger.warning("No Unsplash access key found. Using public API with limitations.")
            return self.scrape_unsplash_public(count)
        
        headers = {
            'Authorization': f'Client-ID {access_key}'
        }
        
        # Topics for diverse real images
        topics = ['nature', 'people', 'architecture', 'street', 'landscape', 'portrait', 'wildlife']
        
        for topic in topics:
            if len(image_urls) >= count:
                break
                
            logger.info(f"Searching Unsplash topic: {topic}")
            
            params = {
                'count': min(30, count - len(image_urls)),
                'topics': topic,
                'orientation': 'all'
            }
            
            try:
                response = self.session.get(base_url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                for photo in data:
                    if len(image_urls) >= count:
                        break
                        
                    # Get regular size URL
                    url = photo['urls']['regular']
                    if url:
                        image_urls.append(url)
                        
                time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
                
            except Exception as e:
                logger.error(f"Error scraping Unsplash topic {topic}: {e}")
                continue
                
        logger.info(f"Found {len(image_urls)} Unsplash images")
        return image_urls[:count]
    
    def scrape_unsplash_public(self, count: int = 100) -> List[str]:
        """Scrape Unsplash using public API (limited)"""
        logger.info("Using Unsplash public API (limited)")
        
        image_urls = []
        base_url = "https://source.unsplash.com"
        
        # Different categories and sizes
        categories = ['nature', 'people', 'architecture', 'street', 'landscape']
        sizes = ['1920x1080', '1280x720', '800x600']
        
        for i in range(count):
            category = random.choice(categories)
            size = random.choice(sizes)
            
            url = f"{base_url}/{size}/?{category}"
            image_urls.append(url)
            
            time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
            
        return image_urls
    
    def scrape_pexels(self, count: int = 100) -> List[str]:
        """Scrape images from Pexels using their API"""
        logger.info(f"Scraping Pexels for {count} images")
        
        image_urls = []
        access_key = os.getenv('PEXELS_API_KEY')
        
        if not access_key:
            logger.warning("No Pexels API key found. Skipping Pexels.")
            return []
        
        base_url = "https://api.pexels.com/v1/search"
        headers = {
            'Authorization': access_key
        }
        
        # Search queries for diverse real images
        queries = ['nature', 'people', 'architecture', 'street', 'landscape', 'portrait', 'wildlife']
        
        for query in queries:
            if len(image_urls) >= count:
                break
                
            logger.info(f"Searching Pexels for: {query}")
            
            params = {
                'query': query,
                'per_page': min(20, count - len(image_urls)),
                'page': 1
            }
            
            try:
                response = self.session.get(base_url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                for photo in data.get('photos', []):
                    if len(image_urls) >= count:
                        break
                        
                    # Get medium size URL
                    url = photo['src']['medium']
                    if url:
                        image_urls.append(url)
                        
                time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
                
            except Exception as e:
                logger.error(f"Error scraping Pexels for {query}: {e}")
                continue
                
        logger.info(f"Found {len(image_urls)} Pexels images")
        return image_urls[:count]
    
    def is_image_file(self, title: str) -> bool:
        """Check if Wikimedia title is an image file"""
        title_lower = title.lower()
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            if title_lower.endswith(ext):
                return True
        return False
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to a valid image"""
        if not url:
            return False
            
        # Check file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            if path.endswith(ext):
                return True
                
        return False
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download image from URL"""
        try:
            response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
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
                
            logger.debug(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def scrape_and_download(self, count: int = 100) -> int:
        """Main method to scrape and download real images"""
        logger.info(f"Starting real image scraping for {count} images")
        
        # Collect URLs from multiple sources
        all_urls = []
        
        # Try different sources
        sources = [
            self.scrape_wikimedia_commons,
            self.scrape_unsplash,
            self.scrape_pexels,
        ]
        
        for source_func in sources:
            try:
                urls = source_func(count // len(sources))
                all_urls.extend(urls)
                if len(all_urls) >= count:
                    break
            except Exception as e:
                logger.error(f"Error with source {source_func.__name__}: {e}")
                continue
        
        # Remove duplicates
        all_urls = list(set(all_urls))
        logger.info(f"Found {len(all_urls)} unique real image URLs")
        
        # Download images
        downloaded = 0
        for i, url in enumerate(all_urls[:count]):
            filename = f"real_image_{i:06d}.jpg"
            
            if self.download_image(url, filename):
                downloaded += 1
                
            # Rate limiting
            time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
            
            if downloaded % 10 == 0:
                logger.info(f"Downloaded {downloaded} images so far...")
        
        logger.info(f"Successfully downloaded {downloaded} real images")
        return downloaded

def main():
    parser = argparse.ArgumentParser(description="Scrape real images")
    parser.add_argument("--count", type=int, default=100, help="Number of images to scrape")
    parser.add_argument("--output", type=str, default=config.REAL_IMAGES_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    scraper = RealImageScraper(args.output)
    downloaded = scraper.scrape_and_download(args.count)
    
    print(f"Successfully downloaded {downloaded} real images to {args.output}")

if __name__ == "__main__":
    main()
