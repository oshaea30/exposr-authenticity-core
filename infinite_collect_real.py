"""
Infinite Real Image Collection Script
Runs continuously until manually stopped (Ctrl+C)
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def get_current_counts():
    """Get current AI and real image counts"""
    ai_dir = Path("data/ai")
    real_dir = Path("data/real")
    
    ai_count = len(list(ai_dir.glob("*.jpg"))) if ai_dir.exists() else 0
    real_count = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
    
    return ai_count, real_count

def print_status(pass_num, ai_count, real_count):
    """Print current status"""
    print(f"\n{'='*70}")
    print(f"🔵 PASS #{pass_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"AI Images:   {ai_count}")
    print(f"Real Images: {real_count}")
    
    if ai_count > 0:
        progress = (real_count / ai_count) * 100
        print(f"Progress:    {progress:.1f}% ({real_count}/{ai_count})")
        
        if real_count < ai_count:
            remaining = ai_count - real_count
            print(f"Remaining:   {remaining} images to reach parity")
        else:
            print(f"Status:      ✅ PARITY ACHIEVED!")
    else:
        print(f"Status:      🎯 Collecting as many as possible!")
    
    print(f"{'='*70}\n")

def run_collection_pass():
    """Run a single collection pass"""
    # Collect a batch of images
    collect_count = 100
    
    print(f"📥 Collecting {collect_count} images from 7 public domain sources...")
    
    # Run the collection script
    cmd = [
        sys.executable,
        "scrape_public_real_images.py",
        "--count", str(collect_count)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print(f"✅ Batch completed successfully!")
            # Show last few lines of output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-8:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"⚠️  Batch completed with some issues")
            error_lines = result.stderr.split('\n')
            for line in error_lines[-5:]:
                if line.strip():
                    print(f"   {line}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Batch timed out after 30 minutes")
        return True
    except Exception as e:
        print(f"❌ Batch error: {e}")
        return True

def main():
    """Main infinite collection loop"""
    print("\n" + "="*70)
    print("🚀 INFINITE REAL IMAGE COLLECTION")
    print("="*70)
    print("This script will run CONTINUOUSLY until you stop it.")
    print("Press Ctrl+C to stop the collection.\n")
    
    pass_num = 1
    start_time = datetime.now()
    
    try:
        while True:  # Infinite loop
            # Get current counts
            ai_count, real_count = get_current_counts()
            
            # Print status
            print_status(pass_num, ai_count, real_count)
            
            # Run collection pass
            run_collection_pass()
            
            # Wait before next pass (30 seconds)
            wait_time = 30
            print(f"\n⏳ Waiting {wait_time} seconds before next pass...")
            print("Press Ctrl+C to stop\n")
            
            time.sleep(wait_time)
            
            pass_num += 1
        
    except KeyboardInterrupt:
        # Final status on interruption
        ai_count, real_count = get_current_counts()
        elapsed = datetime.now() - start_time
        
        print(f"\n\n{'='*70}")
        print("⚠️  COLLECTION STOPPED BY USER")
        print(f"{'='*70}")
        print(f"Total Passes:     {pass_num}")
        print(f"Duration:         {elapsed}")
        print(f"AI Images:        {ai_count}")
        print(f"Real Images:      {real_count}")
        print(f"Total Images:     {ai_count + real_count}")
        
        if ai_count > 0:
            progress = (real_count / ai_count) * 100
            print(f"Progress:         {progress:.1f}%")
        
        print(f"{'='*70}\n")
        print("👋 Goodbye! Run this script again anytime to continue collecting.")

if __name__ == "__main__":
    main()

