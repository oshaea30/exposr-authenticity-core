"""
Upload images to Google Drive using OAuth 2.0 User Flow
For personal use - uploads to 'ExposrDataset' folder in My Drive
"""
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pathlib import Path
import os
import pickle
import csv
from datetime import datetime
from argparse import ArgumentParser

# Scopes for Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Authenticate with Google Drive using OAuth 2.0"""
    creds = None
    token_file = 'token.json'
    
    # Check if we have valid credentials stored
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("🔐 Starting OAuth authentication...")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next time
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print("✅ Credentials saved to token.json")
    
    service = build('drive', 'v3', credentials=creds)
    print("✅ Authenticated with Google Drive!")
    return service

def get_or_create_folder(service, folder_name="ExposrDataset"):
    """Get folder ID, creating folder if it doesn't exist"""
    # Search for existing folder
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    items = results.get('files', [])
    
    if items:
        folder_id = items[0]['id']
        print(f"📁 Found existing folder '{folder_name}' (ID: {folder_id})")
        return folder_id
    else:
        # Create folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"📁 Created folder '{folder_name}' (ID: {folder_id})")
        return folder_id

def file_exists_in_drive(service, filename, folder_id):
    """Check if file already exists in folder"""
    try:
        results = service.files().list(
            q=f"name='{filename}' and parents in '{folder_id}'",
            fields='files(id, name)'
        ).execute()
        return len(results.get('files', [])) > 0
    except:
        return False

def upload_file(service, file_path, file_name, folder_id, skip_existing=True, dry_run=False):
    """Upload a single file to Google Drive"""
    if dry_run:
        print(f"[DRY-RUN] Would upload {file_name} to folder {folder_id}")
        return {'id': 'dry-run-id'}
    
    # Check if file exists
    if skip_existing and file_exists_in_drive(service, file_name, folder_id):
        print(f"⏭️  {file_name} already exists, skipping")
        return None
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(file_path, resumable=True)
    
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"✅ Uploaded {file_name} (ID: {file['id']})")
        return file
    except Exception as e:
        print(f"❌ Failed to upload {file_name}: {e}")
        return None

def log_upload(filename, label, source, file_id, success=True):
    """Log upload to CSV"""
    log_file = 'upload_log.csv'
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'filename', 'label', 'source', 'file_id', 'success'])
        
        writer.writerow([
            datetime.now().isoformat(),
            filename,
            label,
            source,
            file_id,
            success
        ])

def upload_directory(service, directory, folder_id, label, skip_existing=True, dry_run=False):
    """Upload all images from a directory"""
    directory = Path(directory)
    image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.npy"))
    
    if not image_files:
        print(f"📭 No files found in {directory}")
        return 0
    
    print(f"📁 Found {len(image_files)} files to upload from {label}...")
    
    uploaded = 0
    failed = 0
    
    for image_path in image_files:
        file_name = f"{label}_{image_path.name}"
        result = upload_file(service, str(image_path), file_name, folder_id, skip_existing, dry_run)
        
        if result:
            uploaded += 1
            log_upload(file_name, label, str(directory), result.get('id'), success=True)
        else:
            failed += 1
            log_upload(file_name, label, str(directory), '', success=False)
    
    print(f"✅ Uploaded {uploaded}/{len(image_files)} files from {label}")
    return uploaded

def main():
    parser = ArgumentParser(description='Upload Exposr dataset to Google Drive')
    parser.add_argument('--dry-run', action='store_true', help='Test without actually uploading')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip existing files')
    args = parser.parse_args()
    
    print("🚀 Starting Google Drive upload process...")
    print(f"{'🔍 [DRY RUN MODE]' if args.dry_run else '📤 [UPLOAD MODE]'}")
    print("="*70)
    
    # Authenticate
    service = authenticate()
    
    # Get or create folder
    folder_id = get_or_create_folder(service, "ExposrDataset")
    
    # Upload files from each directory
    total_uploaded = 0
    
    for label, directory in [
        ("ai", "data/ai"),
        ("real", "data/real"),
        ("embeddings", "embeddings")
    ]:
        if os.path.exists(directory):
            print(f"\n{'='*70}")
            count = upload_directory(service, directory, folder_id, label, args.skip_existing, args.dry_run)
            total_uploaded += count
    
    print(f"\n{'='*70}")
    print(f"🎉 Upload complete! Total files uploaded: {total_uploaded}")
    print(f"📋 Log saved to upload_log.csv")
    print(f"📁 Drive folder: https://drive.google.com/drive/folders/{folder_id}")

if __name__ == "__main__":
    main()

