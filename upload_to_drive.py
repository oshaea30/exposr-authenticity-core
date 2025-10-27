"""
Upload images to Google Drive shared folder using Service Account
"""
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pathlib import Path
import os

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'exposr-training-e969f8a0541d.json'
FOLDER_ID = '1OIKZGosCqYiv78gH4tswRecD1fwQVpJc'

def authenticate():
    """Authenticate with Google Drive using service account"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    print("✅ Connected to Google Drive using service account!")
    return service

def upload_file_to_drive(service, file_path, file_name, folder_id):
    """Upload a single file to Google Drive"""
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
        print(f"✅ Uploaded {file_name} with ID: {file['id']}")
        return file['id']
    except Exception as e:
        print(f"❌ Failed to upload {file_name}: {e}")
        return None

def upload_images_from_directory(directory, folder_id, label='unknown'):
    """Upload all images from a directory to Drive"""
    service = authenticate()
    
    directory = Path(directory)
    image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
    
    print(f"📁 Found {len(image_files)} images to upload from {label}...")
    
    uploaded = 0
    for image_path in image_files:
        file_name = f"{label}_{image_path.name}"
        result = upload_file_to_drive(service, str(image_path), file_name, folder_id)
        if result:
            uploaded += 1
    
    print(f"\n✅ Successfully uploaded {uploaded}/{len(image_files)} images")
    return uploaded

if __name__ == "__main__":
    # Upload AI images
    print("🤖 Uploading AI images...")
    upload_images_from_directory("data/ai", FOLDER_ID, "ai")
    
    print("\n" + "="*60 + "\n")
    
    # Upload real images
    print("📸 Uploading real images...")
    upload_images_from_directory("data/real", FOLDER_ID, "real")

