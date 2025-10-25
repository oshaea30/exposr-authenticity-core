"""
Playwright Fallback Scraper for Exposr
Handles JavaScript-rendered content when static scrapers fail
"""
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import random
import requests
from datetime import datetime
import csv
from PIL import Image
import hashlib
from playwright.async_api import async_playwright, Browser, Page
import argparse
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class PlaywrightFallbackScraper:
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
        self.min_delay = 2.0
        self.max_delay = 4.0
        
        # Browser settings
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
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
    
    async def setup_browser(self):
        """Setup Playwright browser in headless mode"""
        try:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            self.page = await self.browser.new_page()
            
            # Set user agent
            await self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Set viewport
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            
            logger.info("✅ Playwright browser setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Playwright browser: {e}")
            return False
    
    async def cleanup_browser(self):
        """Cleanup browser resources"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            logger.info("✅ Browser cleanup complete")
        except Exception as e:
            logger.warning(f"Browser cleanup warning: {e}")
    
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
    
    async def scroll_and_load_images(self, scroll_count: int = 5) -> List[str]:
        """Scroll page and extract image URLs"""
        image_urls = []
        
        try:
            for i in range(scroll_count):
                logger.info(f"Scrolling page {i+1}/{scroll_count}")
                
                # Scroll down
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                # Wait for images to load
                await self.page.wait_for_timeout(2000)
                
                # Extract image URLs
                current_images = await self.page.evaluate("""
                    () => {
                        const images = Array.from(document.querySelectorAll('img'));
                        return images.map(img => img.src).filter(src => 
                            src && (src.includes('.jpg') || src.includes('.png') || src.includes('.jpeg'))
                        );
                    }
                """)
                
                image_urls.extend(current_images)
                
                # Remove duplicates
                image_urls = list(set(image_urls))
                
                logger.info(f"Found {len(image_urls)} unique images so far")
                
                # Small delay between scrolls
                await self.page.wait_for_timeout(1000)
            
            return image_urls
            
        except Exception as e:
            logger.error(f"Error during scrolling: {e}")
            return image_urls
    
    async def scrape_lexica_art(self, count: int = 100) -> int:
        """Scrape AI images from Lexica.art using Playwright"""
        logger.info(f"🎨 Scraping Lexica.art with Playwright for {count} AI images")
        
        collected = 0
        
        try:
            # Navigate to Lexica.art
            await self.page.goto("https://lexica.art", wait_until="networkidle", timeout=30000)
            logger.info("✅ Loaded Lexica.art homepage")
            
            # Wait for page to load
            await self.page.wait_for_timeout(3000)
            
            # Scroll and load images
            image_urls = await self.scroll_and_load_images(scroll_count=5)
            
            logger.info(f"Found {len(image_urls)} potential images from Lexica.art")
            
            # Download images
            for i, img_url in enumerate(image_urls):
                if collected >= count:
                    break
                
                try:
                    # Skip invalid URLs
                    if not img_url or 'data:' in img_url or 'placeholder' in img_url.lower():
                        continue
                    
                    filename = f"lexica_playwright_{collected:06d}.jpg"
                    
                    if self.download_image(img_url, filename):
                        file_size = os.path.getsize(self.output_dir / filename)
                        self.log_image_metadata(filename, "lexica_art_playwright", "public", img_url, file_size)
                        collected += 1
                        logger.info(f"✅ Downloaded Lexica image {collected}/{count}")
                    
                    self.rate_limit()
                    
                except Exception as e:
                    logger.warning(f"Skipped image {img_url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error scraping Lexica.art: {e}")
        
        logger.info(f"Collected {collected} images from Lexica.art (Playwright)")
        return collected
    
    async def scrape_artbreeder(self, count: int = 100) -> int:
        """Scrape AI images from Artbreeder using Playwright"""
        logger.info(f"🎨 Scraping Artbreeder with Playwright for {count} AI images")
        
        collected = 0
        
        try:
            # Navigate to Artbreeder browse page
            await self.page.goto("https://www.artbreeder.com/browse", wait_until="networkidle", timeout=30000)
            logger.info("✅ Loaded Artbreeder browse page")
            
            # Wait for page to load
            await self.page.wait_for_timeout(3000)
            
            # Try to dismiss any popups or modals
            try:
                await self.page.click('button[aria-label="Close"]', timeout=2000)
            except:
                pass
            
            # Scroll and load images
            image_urls = await self.scroll_and_load_images(scroll_count=5)
            
            logger.info(f"Found {len(image_urls)} potential images from Artbreeder")
            
            # Download images
            for i, img_url in enumerate(image_urls):
                if collected >= count:
                    break
                
                try:
                    # Skip invalid URLs
                    if not img_url or 'data:' in img_url or 'placeholder' in img_url.lower():
                        continue
                    
                    filename = f"artbreeder_playwright_{collected:06d}.jpg"
                    
                    if self.download_image(img_url, filename):
                        file_size = os.path.getsize(self.output_dir / filename)
                        self.log_image_metadata(filename, "artbreeder_playwright", "public", img_url, file_size)
                        collected += 1
                        logger.info(f"✅ Downloaded Artbreeder image {collected}/{count}")
                    
                    self.rate_limit()
                    
                except Exception as e:
                    logger.warning(f"Skipped image {img_url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error scraping Artbreeder: {e}")
        
        logger.info(f"Collected {collected} images from Artbreeder (Playwright)")
        return collected
    
    async def scrape_all_sources(self, target_count: int = 1000) -> int:
        """Scrape from all AI image sources using Playwright"""
        logger.info(f"🚀 Starting Playwright AI image collection for {target_count} images")
        
        total_collected = 0
        
        # Distribute target across sources
        per_source = target_count // 2
        
        # Lexica.art
        lexica_count = await self.scrape_lexica_art(per_source)
        total_collected += lexica_count
        
        # Artbreeder
        artbreeder_count = await self.scrape_artbreeder(per_source)
        total_collected += artbreeder_count
        
        logger.info(f"Total collected with Playwright: {total_collected} AI images")
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

async def main():
    """Main async function to run Playwright scraper"""
    parser = argparse.ArgumentParser(description="Playwright fallback scraper for AI images")
    parser.add_argument("--count", type=int, default=100, help="Target number of images to collect")
    parser.add_argument("--output", type=str, default="data/ai", help="Output directory")
    parser.add_argument("--lexica-only", action="store_true", help="Only scrape Lexica.art")
    parser.add_argument("--artbreeder-only", action="store_true", help="Only scrape Artbreeder")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    scraper = PlaywrightFallbackScraper(args.output)
    
    if args.stats:
        stats = scraper.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Setup browser
    if not await scraper.setup_browser():
        print("❌ Failed to setup browser")
        return
    
    try:
        if args.lexica_only:
            collected = await scraper.scrape_lexica_art(args.count)
        elif args.artbreeder_only:
            collected = await scraper.scrape_artbreeder(args.count)
        else:
            collected = await scraper.scrape_all_sources(args.count)
        
        print(f"✅ Successfully collected {collected} AI-generated images with Playwright")
        print(f"📁 Images saved to {args.output}")
        print(f"📋 Metadata logged to dataset_log.csv")
        
    finally:
        await scraper.cleanup_browser()

if __name__ == "__main__":
    asyncio.run(main())
