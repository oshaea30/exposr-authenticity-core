# Google Drive Integration Setup

## Overview
Complete OAuth 2.0 integration for syncing files from Google Drive to local storage for Exposr training pipeline.

## Setup Instructions

### 1. Place Credentials File
Copy your `credentials.json` (from Google Cloud Console) to:
```
auth/oauth_client_secret.json
```

### 2. Install Dependencies
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 3. Test Authentication
```bash
python scripts/test_drive_sync.py
```

First run will open a browser for OAuth authorization. Credentials will be saved to `auth/token.json` for future use.

## Usage

### Sync Files from Drive Folder
```bash
# Sync all files from a specific folder
python scripts/sync_drive.py FOLDER_ID

# Sync with limit
python scripts/sync_drive.py FOLDER_ID --limit 10
```

### Find Folder ID
1. Open Google Drive
2. Navigate to desired folder
3. Copy ID from URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`

### Files are saved to:
- Images: `training_data/google_drive_sync/images/`
- Videos: `training_data/google_drive_sync/videos/`
- PDFs: `training_data/google_drive_sync/documents/`

### Metadata Log
All downloads are logged to: `gdrive_sync_log.csv`

## Features

✅ OAuth 2.0 authentication with auto-refresh  
✅ Download images (JPG, PNG), videos (MP4), documents (PDF)  
✅ Automatic post-processing for images  
✅ Skip existing files to avoid duplicates  
✅ Complete metadata logging  

## File Structure
```
/auth/
  - oauth_client_secret.json (your credentials)
  - token.json (auto-generated, don't commit!)

/scripts/
  - sync_drive.py (main sync script)
  - test_drive_sync.py (test script)
  - process_synced_files.py (post-processing)

/training_data/
  - google_drive_sync/
    - images/
    - videos/
    - documents/

gdrive_sync_log.csv (download metadata)
```

