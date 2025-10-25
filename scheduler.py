"""
Training Scheduler for Automated Model Retraining
"""
import os
import time
import schedule
import logging
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TrainingScheduler:
    def __init__(self, log_file: str = "training_scheduler.log"):
        self.log_file = log_file
        self.training_history = []
        self.is_running = False
        
    def log_training_event(self, event_type: str, message: str, success: bool = True):
        """Log training events"""
        timestamp = datetime.now().isoformat()
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message,
            'success': success
        }
        
        self.training_history.append(event)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {event_type}: {message} (Success: {success})\n")
        
        logger.info(f"{event_type}: {message}")
    
    def check_data_availability(self) -> bool:
        """Check if enough data is available for training"""
        ai_dir = Path(config.AI_IMAGES_DIR)
        real_dir = Path(config.REAL_IMAGES_DIR)
        
        ai_count = len(list(ai_dir.glob("*"))) if ai_dir.exists() else 0
        real_count = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
        
        min_required = config.MIN_IMAGES_PER_CLASS
        
        if ai_count < min_required or real_count < min_required:
            self.log_training_event(
                "DATA_CHECK", 
                f"Insufficient data: AI={ai_count}, Real={real_count} (min={min_required})", 
                False
            )
            return False
        
        self.log_training_event(
            "DATA_CHECK", 
            f"Data available: AI={ai_count}, Real={real_count}", 
            True
        )
        return True
    
    def scrape_new_data(self) -> bool:
        """Scrape new data for training"""
        try:
            self.log_training_event("SCRAPING", "Starting data scraping", True)
            
            # Scrape AI images
            ai_cmd = [
                "python", "scraper/scrape_ai.py", 
                "--count", str(config.MIN_IMAGES_PER_CLASS)
            ]
            result = subprocess.run(ai_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                self.log_training_event("SCRAPING", f"AI scraping failed: {result.stderr}", False)
                return False
            
            # Scrape real images
            real_cmd = [
                "python", "scraper/scrape_real.py", 
                "--count", str(config.MIN_IMAGES_PER_CLASS)
            ]
            result = subprocess.run(real_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                self.log_training_event("SCRAPING", f"Real scraping failed: {result.stderr}", False)
                return False
            
            self.log_training_event("SCRAPING", "Data scraping completed", True)
            return True
            
        except subprocess.TimeoutExpired:
            self.log_training_event("SCRAPING", "Scraping timed out", False)
            return False
        except Exception as e:
            self.log_training_event("SCRAPING", f"Scraping error: {str(e)}", False)
            return False
    
    def generate_embeddings(self) -> bool:
        """Generate embeddings for training data"""
        try:
            self.log_training_event("EMBEDDINGS", "Starting embedding generation", True)
            
            cmd = [
                "python", "generate_embeddings.py",
                "--ai-dir", config.AI_IMAGES_DIR,
                "--real-dir", config.REAL_IMAGES_DIR,
                "--output", os.path.join(config.EMBEDDINGS_DIR, "image_vectors.npy")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode != 0:
                self.log_training_event("EMBEDDINGS", f"Embedding generation failed: {result.stderr}", False)
                return False
            
            self.log_training_event("EMBEDDINGS", "Embedding generation completed", True)
            return True
            
        except subprocess.TimeoutExpired:
            self.log_training_event("EMBEDDINGS", "Embedding generation timed out", False)
            return False
        except Exception as e:
            self.log_training_event("EMBEDDINGS", f"Embedding generation error: {str(e)}", False)
            return False
    
    def train_model(self) -> bool:
        """Train the classifier model"""
        try:
            self.log_training_event("TRAINING", "Starting model training", True)
            
            cmd = [
                "python", "train.py",
                "--embeddings", os.path.join(config.EMBEDDINGS_DIR, "image_vectors.npy"),
                "--output", os.path.join(config.MODELS_DIR, "classifier.pkl"),
                "--classifier", config.CLASSIFIER_TYPE
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
            
            if result.returncode != 0:
                self.log_training_event("TRAINING", f"Model training failed: {result.stderr}", False)
                return False
            
            # Extract accuracy from output
            output_lines = result.stdout.split('\n')
            accuracy = None
            for line in output_lines:
                if "Test accuracy:" in line:
                    try:
                        accuracy = float(line.split(":")[1].strip())
                        break
                    except:
                        pass
            
            if accuracy:
                self.log_training_event("TRAINING", f"Model training completed with accuracy: {accuracy:.4f}", True)
            else:
                self.log_training_event("TRAINING", "Model training completed", True)
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log_training_event("TRAINING", "Model training timed out", False)
            return False
        except Exception as e:
            self.log_training_event("TRAINING", f"Model training error: {str(e)}", False)
            return False
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        if self.is_running:
            self.log_training_event("SCHEDULER", "Training already in progress, skipping", False)
            return
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            self.log_training_event("SCHEDULER", "Starting scheduled training pipeline", True)
            
            # Check if we need to scrape new data
            if not self.check_data_availability():
                self.log_training_event("SCHEDULER", "Scraping new data", True)
                if not self.scrape_new_data():
                    self.log_training_event("SCHEDULER", "Failed to scrape new data", False)
                    return
            
            # Generate embeddings
            if not self.generate_embeddings():
                self.log_training_event("SCHEDULER", "Failed to generate embeddings", False)
                return
            
            # Train model
            if not self.train_model():
                self.log_training_event("SCHEDULER", "Failed to train model", False)
                return
            
            # Calculate total time
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.log_training_event("SCHEDULER", f"Training pipeline completed successfully in {duration}", True)
            
        except Exception as e:
            self.log_training_event("SCHEDULER", f"Training pipeline error: {str(e)}", False)
        finally:
            self.is_running = False
    
    def setup_schedule(self):
        """Setup the training schedule"""
        if config.TRAINING_FREQUENCY == "daily":
            schedule.every().day.at(config.TRAINING_TIME).do(self.run_training_pipeline)
            logger.info(f"Scheduled daily training at {config.TRAINING_TIME}")
        elif config.TRAINING_FREQUENCY == "weekly":
            # Default to Monday at the specified time
            schedule.every().monday.at(config.TRAINING_TIME).do(self.run_training_pipeline)
            logger.info(f"Scheduled weekly training on Mondays at {config.TRAINING_TIME}")
        else:
            logger.error(f"Unknown training frequency: {config.TRAINING_FREQUENCY}")
            return False
        
        return True
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Starting training scheduler")
        
        if not self.setup_schedule():
            return
        
        # Log next scheduled run
        next_run = schedule.next_run()
        if next_run:
            self.log_training_event("SCHEDULER", f"Next training scheduled for: {next_run}", True)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.log_training_event("SCHEDULER", "Scheduler stopped by user", True)
        except Exception as e:
            self.log_training_event("SCHEDULER", f"Scheduler error: {str(e)}", False)
    
    def get_training_history(self) -> list:
        """Get training history"""
        return self.training_history
    
    def save_history(self, filename: str = "training_history.json"):
        """Save training history to file"""
        with open(filename, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Training Scheduler for AI/Real Image Detection")
    parser.add_argument("--run-once", action="store_true", help="Run training pipeline once and exit")
    parser.add_argument("--frequency", type=str, choices=["daily", "weekly"], help="Override training frequency")
    parser.add_argument("--time", type=str, help="Override training time (HH:MM format)")
    parser.add_argument("--log-file", type=str, default="training_scheduler.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.frequency:
        config.TRAINING_FREQUENCY = args.frequency
    if args.time:
        config.TRAINING_TIME = args.time
    
    scheduler = TrainingScheduler(args.log_file)
    
    if args.run_once:
        print("Running training pipeline once...")
        scheduler.run_training_pipeline()
        scheduler.save_history()
    else:
        print(f"Starting scheduler with {config.TRAINING_FREQUENCY} training at {config.TRAINING_TIME}")
        scheduler.run_scheduler()

if __name__ == "__main__":
    main()
