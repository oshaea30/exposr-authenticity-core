"""
AI Image Scraper for Midjourney and other AI-generated images
"""
import os
import requests
import time
import random
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
import argparse
from typing import List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class AIMageScraper:
    def __init__(self, output_dir: str = config.AI_IMAGES_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome driver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def scrape_midjourney_gallery(self, count: int = 100) -> List[str]:
        """
        Scrape images from Midjourney gallery websites
        Note: This is a simplified approach. Real implementation would need to handle
        authentication and rate limiting for Midjourney's official gallery.
        """
        logger.info(f"Starting Midjourney gallery scraping for {count} images")
        
        # Alternative sources for AI-generated images
        sources = [
            "https://www.lexica.art/",  # AI art search engine
            "https://www.krea.ai/gallery",  # AI art gallery
            "https://www.artbreeder.com/browse",  # AI art platform
        ]
        
        image_urls = []
        driver = self.setup_driver()
        
        try:
            for source in sources:
                if len(image_urls) >= count:
                    break
                    
                logger.info(f"Scraping from {source}")
                driver.get(source)
                time.sleep(random.uniform(2, 4))
                
                # Wait for images to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "img"))
                )
                
                # Find all image elements
                images = driver.find_elements(By.TAG_NAME, "img")
                
                for img in images:
                    if len(image_urls) >= count:
                        break
                        
                    try:
                        src = img.get_attribute("src")
                        if src and self.is_valid_image_url(src):
                            image_urls.append(src)
                            logger.debug(f"Found image: {src}")
                    except Exception as e:
                        logger.debug(f"Error processing image: {e}")
                        continue
                        
                time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
                
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
        finally:
            driver.quit()
            
        logger.info(f"Found {len(image_urls)} AI-generated image URLs")
        return image_urls[:count]
    
    def scrape_lexica_art(self, count: int = 100) -> List[str]:
        """Scrape from Lexica.art - AI art search engine"""
        logger.info(f"Scraping Lexica.art for {count} images")
        
        image_urls = []
        driver = self.setup_driver()
        
        try:
            # Navigate to Lexica
            driver.get("https://www.lexica.art/")
            time.sleep(3)
            
            # Scroll to load more images
            for _ in range(5):  # Scroll 5 times
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Find image elements
                images = driver.find_elements(By.CSS_SELECTOR, "img[src*='lexica-s3']")
                
                for img in images:
                    if len(image_urls) >= count:
                        break
                        
                    try:
                        src = img.get_attribute("src")
                        if src and self.is_valid_image_url(src):
                            image_urls.append(src)
                    except Exception as e:
                        logger.debug(f"Error processing image: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error scraping Lexica: {e}")
        finally:
            driver.quit()
            
        return image_urls[:count]
    
    def scrape_artbreeder(self, count: int = 100) -> List[str]:
        """Scrape from Artbreeder - AI art platform"""
        logger.info(f"Scraping Artbreeder for {count} images")
        
        image_urls = []
        driver = self.setup_driver()
        
        try:
            driver.get("https://www.artbreeder.com/browse")
            time.sleep(3)
            
            # Scroll to load images
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                images = driver.find_elements(By.CSS_SELECTOR, "img[src*='artbreeder']")
                
                for img in images:
                    if len(image_urls) >= count:
                        break
                        
                    try:
                        src = img.get_attribute("src")
                        if src and self.is_valid_image_url(src):
                            image_urls.append(src)
                    except Exception as e:
                        logger.debug(f"Error processing image: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error scraping Artbreeder: {e}")
        finally:
            driver.quit()
            
        return image_urls[:count]
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to a valid image"""
        if not url:
            return False
            
        # Check if it's a data URL or relative URL
        if url.startswith('data:') or url.startswith('/'):
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
        """Main method to scrape and download AI images"""
        logger.info(f"Starting AI image scraping for {count} images")
        
        # Collect URLs from multiple sources
        all_urls = []
        
        # Try different sources
        sources = [
            self.scrape_lexica_art,
            self.scrape_artbreeder,
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
        logger.info(f"Found {len(all_urls)} unique AI image URLs")
        
        # Download images
        downloaded = 0
        for i, url in enumerate(all_urls[:count]):
            filename = f"ai_image_{i:06d}.jpg"
            
            if self.download_image(url, filename):
                downloaded += 1
                
            # Rate limiting
            time.sleep(random.uniform(config.SCRAPE_DELAY_MIN, config.SCRAPE_DELAY_MAX))
            
            if downloaded % 10 == 0:
                logger.info(f"Downloaded {downloaded} images so far...")
        
        logger.info(f"Successfully downloaded {downloaded} AI images")
        return downloaded

def main():
    parser = argparse.ArgumentParser(description="Scrape AI-generated images")
    parser.add_argument("--count", type=int, default=100, help="Number of images to scrape")
    parser.add_argument("--output", type=str, default=config.AI_IMAGES_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    scraper = AIMageScraper(args.output)
    downloaded = scraper.scrape_and_download(args.count)
    
    print(f"Successfully downloaded {downloaded} AI images to {args.output}")

if __name__ == "__main__":
    main()
