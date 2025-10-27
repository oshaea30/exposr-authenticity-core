"""
Test script for Google Drive sync functionality
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sync_drive import DriveSync

def main():
    """Test Drive sync with example folder"""
    sync = DriveSync()
    
    # Authenticate
    print("🔐 Authenticating...")
    if not sync.authenticate():
        print("❌ Authentication failed")
        return
    
    # Example folder ID - replace with your actual folder ID
    folder_id = input("Enter Google Drive folder ID (or press Enter to test with sample folder): ")
    
    if not folder_id:
        print("⚠️  No folder ID provided. Exiting.")
        return
    
    # List files
    print(f"\n📂 Listing files in folder {folder_id}...")
    files = sync.list_files_in_folder(folder_id)
    
    if not files:
        print("📭 No files found in folder")
        return
    
    print(f"\n📊 Found {len(files)} files:")
    for i, file in enumerate(files[:10], 1):  # Show first 10
        size = int(file.get('size', 0)) if file.get('size') else 0
        size_str = f"{size / 1024:.1f}KB" if size < 1024*1024 else f"{size / (1024*1024):.1f}MB"
        print(f"  {i}. {file['name']} ({file['mimeType']}) - {size_str}")
    
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    # Download first 5 supported files
    print("\n⬇️  Downloading first 5 supported files...")
    supported_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.pdf'}
    supported_files = [
        f for f in files 
        if any(f['name'].lower().endswith(ext) for ext in supported_extensions)
    ][:5]
    
    for file in supported_files:
        filename = file['name']
        file_type = Path(filename).suffix.lower()
        sync.download_file(file['id'], filename, file_type)
    
    # Show log
    log_file = 'gdrive_sync_log.csv'
    print(f"\n📋 Checking log file: {log_file}")
    if os.path.exists(log_file):
        print("Recent downloads:")
        with open(log_file, 'r') as f:
            lines = f.readlines()[-6:]  # Last 6 lines
            for line in lines:
                print(f"  {line.strip()}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    from pathlib import Path
    import os
    main()

