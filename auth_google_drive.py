"""
Authenticate with Google Drive once
Run this script to initialize your OAuth token
"""
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Authenticate with Google Drive"""
    creds = None
    token_file = 'token.pickle'
    
    # Check if we already have credentials
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
            print("✅ Using existing credentials")
    
    # If there are no valid credentials, request authorization
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing credentials...")
            creds.refresh(Request())
        else:
            print("🔐 Starting authentication flow...")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
            print("✅ Credentials saved to token.pickle")
    
    # Build the Drive service
    service = build('drive', 'v3', credentials=creds)
    print("✅ Connected to Google Drive!")
    
    return service

if __name__ == "__main__":
    service = authenticate()
    print("\n🎉 Authentication complete!")
    print("You can now upload images to Google Drive.")

