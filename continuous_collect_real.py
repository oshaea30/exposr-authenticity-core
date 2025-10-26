"""
Continuous Real Image Collection Script
Runs until reaching parity with AI images (178 real images)
"""
import subprocess
import sys
import time
from pathlib import Path

def get_current_counts():
    """Get current AI and real image counts"""
    ai_dir = Path("data/ai")
    real_dir = Path("data/real")
    
    ai_count = len(list(ai_dir.glob("*.jpg"))) if ai_dir.exists() else 0
    real_count = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
    
    return ai_count, real_count

def run_collection_pass(pass_num, target_real):
    """Run a single collection pass"""
    print(f"\n{'='*60}")
    print(f"🔵 Collection Pass #{pass_num}")
    print(f"{'='*60}\n")
    
    ai_count, real_count = get_current_counts()
    remaining = target_real - real_count
    
    if remaining <= 0:
        print(f"✅ Target achieved! Real images: {real_count} >= AI images: {ai_count}")
        return False
    
    # Collect more than needed to account for failures
    collect_count = min(remaining + 50, 150)
    
    print(f"📊 Current Status:")
    print(f"   AI Images: {ai_count}")
    print(f"   Real Images: {real_count}")
    print(f"   Remaining: {remaining} images")
    print(f"   Collecting: {collect_count} images this pass\n")
    
    # Run the collection script
    cmd = [
        sys.executable,
        "scrape_public_real_images.py",
        "--count", str(collect_count)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print(f"✅ Pass #{pass_num} completed successfully!")
            print(result.stdout[-500:])  # Last 500 chars of output
        else:
            print(f"⚠️  Pass #{pass_num} completed with some issues")
            print(result.stderr[-500:])
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Pass #{pass_num} timed out after 30 minutes")
        return True
    except Exception as e:
        print(f"❌ Pass #{pass_num} error: {e}")
        return True

def main():
    """Main continuous collection loop"""
    print("🚀 Starting Continuous Real Image Collection")
    print("Target: Parity with AI images (178 real images)")
    print("Press Ctrl+C to stop\n")
    
    # Get target from AI images
    ai_count, real_count = get_current_counts()
    
    if ai_count == 0:
        print("❌ No AI images found. Please run AI collection first.")
        return
    
    target_real = ai_count
    print(f"📊 Baseline: {ai_count} AI images")
    print(f"📊 Current: {real_count} real images")
    print(f"📊 Target: {target_real} real images\n")
    
    pass_num = 1
    max_passes = 10
    
    try:
        while pass_num <= max_passes:
            success = run_collection_pass(pass_num, target_real)
            
            if not success:
                break
            
            # Check if we've reached target
            _, current_real = get_current_counts()
            
            if current_real >= target_real:
                print(f"\n🎉 Target achieved! {current_real} real images >= {target_real} target")
                break
            
            # Wait before next pass
            if pass_num < max_passes:
                wait_time = 60  # 1 minute between passes
                print(f"\n⏳ Waiting {wait_time} seconds before next pass...")
                time.sleep(wait_time)
            
            pass_num += 1
        
        # Final status
        ai_count, real_count = get_current_counts()
        
        print(f"\n{'='*60}")
        print(f"🏁 Collection Complete!")
        print(f"{'='*60}")
        print(f"AI Images: {ai_count}")
        print(f"Real Images: {real_count}")
        print(f"Total Images: {ai_count + real_count}")
        print(f"Parity: {'✅ Achieved' if real_count >= ai_count else '❌ Not reached'}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        ai_count, real_count = get_current_counts()
        print(f"\n\n⚠️  Collection interrupted by user")
        print(f"Final status: {real_count} real images / {ai_count} AI images")
        print(f"Progress: {(real_count/ai_count*100):.1f}%")

if __name__ == "__main__":
    main()

