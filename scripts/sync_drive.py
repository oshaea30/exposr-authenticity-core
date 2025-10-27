"""
Sync files from Google Drive to local storage
Handles OAuth 2.0 authentication, downloads, and post-processing
"""
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from pathlib import Path
import csv
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'auth/oauth_client_secret.json'
TOKEN_FILE = 'auth/token.json'
SYNC_DIR = 'training_data/google_drive_sync/'
LOG_FILE = 'gdrive_sync_log.csv'

# Supported file types
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.mp4', '.pdf'}

class DriveSync:
    def __init__(self):
        self.service = None
        self.sync_dir = Path(SYNC_DIR)
        self.sync_dir.mkdir(parents=True, exist_ok=True)
    
    def authenticate(self) -> bool:
        """Authenticate with Google Drive using OAuth 2.0"""
        creds = None
        
        # Load existing token
        if os.path.exists(TOKEN_FILE):
            logger.info("Loading existing credentials...")
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials...")
            try:
                creds.refresh(Request())
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
                logger.info("✅ Credentials refreshed")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                creds = None
        
        # Need new authentication
        if not creds or not creds.valid:
            logger.info("Starting OAuth flow...")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
                logger.info("✅ Credentials saved")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return False
        
        # Build service
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("✅ Authenticated with Google Drive!")
        return True
    
    def list_files_in_folder(self, folder_id: str) -> List[Dict]:
        """List all files in a folder"""
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        logger.info(f"📂 Listing files in folder {folder_id}...")
        
        results = []
        page_token = None
        
        while True:
            try:
                response = self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    pageSize=1000,
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageToken=page_token
                ).execute()
                
                files = response.get('files', [])
                results.extend(files)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                break
        
        logger.info(f"📊 Found {len(results)} files")
        return results
    
    def download_file(self, file_id: str, filename: str, file_type: str) -> bool:
        """Download a file from Google Drive"""
        if not self.service:
            return False
        
        # Determine local path based on file type
        if file_type in {'.jpg', '.jpeg', '.png'}:
            subdir = 'images'
        elif file_type == '.mp4':
            subdir = 'videos'
        elif file_type == '.pdf':
            subdir = 'documents'
        else:
            subdir = 'other'
        
        local_dir = self.sync_dir / subdir
        local_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = local_dir / filename
        
        # Skip if file exists
        if local_path.exists():
            logger.info(f"⏭️  {filename} already exists, skipping")
            return True
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_data = io.BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"⬇️  Downloading {filename}: {int(status.progress() * 100)}%")
            
            # Save file
            with open(local_path, 'wb') as f:
                f.write(file_data.getvalue())
            
            logger.info(f"✅ Downloaded {filename} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def log_download(self, filename: str, file_id: str, file_type: str, success: bool):
        """Log download to CSV"""
        file_exists = os.path.exists(LOG_FILE)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'filename', 'drive_id', 'file_type', 'success'])
            
            writer.writerow([
                datetime.now().isoformat(),
                filename,
                file_id,
                file_type,
                success
            ])
    
    def sync_folder(self, folder_id: str, limit: Optional[int] = None) -> int:
        """Sync all files from a folder"""
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        files = self.list_files_in_folder(folder_id)
        
        # Filter by supported extensions
        supported_files = [
            f for f in files 
            if any(f['name'].lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        ]
        
        if limit:
            supported_files = supported_files[:limit]
        
        logger.info(f"🔄 Syncing {len(supported_files)} supported files...")
        
        downloaded = 0
        for file in supported_files:
            filename = file['name']
            
            # Determine file type
            file_type = Path(filename).suffix.lower()
            
            success = self.download_file(file['id'], filename, file_type)
            self.log_download(filename, file['id'], file_type, success)
            
            if success:
                downloaded += 1
        
        logger.info(f"✅ Downloaded {downloaded}/{len(supported_files)} files")
        return downloaded

def main():
    """Main sync function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync files from Google Drive')
    parser.add_argument('folder_id', help='Google Drive folder ID')
    parser.add_argument('--limit', type=int, help='Limit number of files to download')
    
    args = parser.parse_args()
    
    sync = DriveSync()
    
    # Authenticate
    if not sync.authenticate():
        logger.error("❌ Authentication failed")
        return
    
    # Sync folder
    sync.sync_folder(args.folder_id, args.limit)
    logger.info("🎉 Sync complete!")

if __name__ == "__main__":
    main()

