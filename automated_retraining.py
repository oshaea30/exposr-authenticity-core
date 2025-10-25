"""
Automated Retraining System with Performance Tracking
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
import numpy as np
import config
from enhanced_data_collection import EnhancedDataCollector
from embed_images import EmbeddingPipeline
from enhanced_train import EnhancedClassifierTrainer

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class AutomatedRetrainingSystem:
    def __init__(self, log_file: str = "retraining_system.log"):
        self.log_file = log_file
        self.training_history = []
        self.performance_tracking = []
        self.is_running = False
        self.version_counter = 1
        
        # Load existing history
        self.load_training_history()
        self.load_performance_tracking()
        
    def load_training_history(self):
        """Load training history from file"""
        history_file = Path("training_reports/retraining_history.json")
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.training_history = json.load(f)
        else:
            self.training_history = []
    
    def save_training_history(self):
        """Save training history to file"""
        history_file = Path("training_reports/retraining_history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_performance_tracking(self):
        """Load performance tracking data"""
        tracking_file = Path("training_reports/performance_tracking.json")
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                self.performance_tracking = json.load(f)
        else:
            self.performance_tracking = []
    
    def save_performance_tracking(self):
        """Save performance tracking data"""
        tracking_file = Path("training_reports/performance_tracking.json")
        tracking_file.parent.mkdir(exist_ok=True)
        
        with open(tracking_file, 'w') as f:
            json.dump(self.performance_tracking, f, indent=2)
    
    def log_retraining_event(self, event_type: str, message: str, success: bool = True, data: Dict[str, Any] = None):
        """Log retraining events with structured data"""
        timestamp = datetime.now().isoformat()
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message,
            'success': success,
            'data': data or {}
        }
        
        self.training_history.append(event)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {event_type}: {message} (Success: {success})\n")
        
        logger.info(f"{event_type}: {message}")
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check data quality and availability"""
        ai_collector = EnhancedDataCollector(config.AI_IMAGES_DIR, "ai")
        real_collector = EnhancedDataCollector(config.REAL_IMAGES_DIR, "real")
        
        ai_stats = ai_collector.get_dataset_stats()
        real_stats = real_collector.get_dataset_stats()
        
        quality_report = {
            'ai_images': ai_stats['total_images'],
            'real_images': real_stats['total_images'],
            'total_images': ai_stats['total_images'] + real_stats['total_images'],
            'ai_size_mb': ai_stats['total_size_mb'],
            'real_size_mb': real_stats['total_size_mb'],
            'min_required': config.MIN_IMAGES_PER_CLASS,
            'sufficient_data': ai_stats['total_images'] >= config.MIN_IMAGES_PER_CLASS and real_stats['total_images'] >= config.MIN_IMAGES_PER_CLASS,
            'last_ai_update': ai_stats['last_updated'],
            'last_real_update': real_stats['last_updated']
        }
        
        return quality_report
    
    def collect_new_data(self, target_count: int = None) -> bool:
        """Collect new data if needed"""
        try:
            quality_report = self.check_data_quality()
            
            if quality_report['sufficient_data']:
                self.log_retraining_event("DATA_CHECK", "Sufficient data available", True, quality_report)
                return True
            
            # Determine target count
            if target_count is None:
                target_count = max(config.MIN_IMAGES_PER_CLASS, quality_report['min_required'])
            
            self.log_retraining_event("DATA_COLLECTION", f"Starting data collection for {target_count} images per class", True)
            
            # Collect AI images
            ai_collector = EnhancedDataCollector(config.AI_IMAGES_DIR, "ai")
            ai_collected = ai_collector.scrape_ai_sources(target_count)
            
            # Collect real images
            real_collector = EnhancedDataCollector(config.REAL_IMAGES_DIR, "real")
            real_collected = real_collector.scrape_real_sources(target_count)
            
            # Check if we have enough data now
            final_quality = self.check_data_quality()
            
            if final_quality['sufficient_data']:
                self.log_retraining_event("DATA_COLLECTION", f"Data collection completed: AI={ai_collected}, Real={real_collected}", True, final_quality)
                return True
            else:
                self.log_retraining_event("DATA_COLLECTION", "Insufficient data after collection", False, final_quality)
                return False
                
        except Exception as e:
            self.log_retraining_event("DATA_COLLECTION", f"Data collection failed: {str(e)}", False)
            return False
    
    def generate_embeddings(self, version: str = None) -> bool:
        """Generate embeddings with version tracking"""
        try:
            if version is None:
                version = f"1.{self.version_counter}"
            
            self.log_retraining_event("EMBEDDING_GENERATION", f"Starting embedding generation v{version}", True)
            
            # Initialize embedding pipeline
            pipeline = EmbeddingPipeline(version=version)
            
            # Process dataset
            ai_embeddings, real_embeddings, labels, processing_metadata = pipeline.process_dataset(
                config.AI_IMAGES_DIR, config.REAL_IMAGES_DIR
            )
            
            # Validate embeddings
            all_embeddings = np.vstack([ai_embeddings, real_embeddings]) if len(ai_embeddings) > 0 and len(real_embeddings) > 0 else np.array([])
            
            if not pipeline.validate_embeddings(all_embeddings):
                self.log_retraining_event("EMBEDDING_GENERATION", "Embedding validation failed", False)
                return False
            
            # Save embeddings
            output_path, metadata_path = pipeline.save_embeddings_with_version(
                ai_embeddings, real_embeddings, labels, processing_metadata
            )
            
            self.log_retraining_event("EMBEDDING_GENERATION", f"Embedding generation completed v{version}", True, {
                'ai_embeddings': len(ai_embeddings),
                'real_embeddings': len(real_embeddings),
                'output_path': str(output_path),
                'metadata_path': str(metadata_path)
            })
            
            return True
            
        except Exception as e:
            self.log_retraining_event("EMBEDDING_GENERATION", f"Embedding generation failed: {str(e)}", False)
            return False
    
    def train_model(self, embeddings_path: str, version: str = None) -> bool:
        """Train model with enhanced tracking"""
        try:
            if version is None:
                version = f"1.{self.version_counter}"
            
            self.log_retraining_event("MODEL_TRAINING", f"Starting model training v{version}", True)
            
            # Initialize trainer
            trainer = EnhancedClassifierTrainer(version=version)
            
            # Train model
            training_record = trainer.train_with_validation(embeddings_path, optimize=True)
            
            # Save model
            model_path = Path(config.MODELS_DIR) / "classifier.pkl"
            trainer.save_model_with_version(str(model_path))
            
            # Track performance
            performance_data = {
                'version': version,
                'timestamp': training_record['timestamp'],
                'test_accuracy': training_record['performance']['test_accuracy'],
                'cv_mean': training_record['performance']['cv_mean'],
                'cv_std': training_record['performance']['cv_std'],
                'roc_auc': training_record['performance'].get('roc_auc'),
                'classifier_type': training_record['classifier_type'],
                'best_params': training_record['best_params']
            }
            
            self.performance_tracking.append(performance_data)
            self.save_performance_tracking()
            
            # Check for performance improvement
            improvement = self.check_performance_improvement(performance_data)
            
            self.log_retraining_event("MODEL_TRAINING", f"Model training completed v{version}", True, {
                'test_accuracy': training_record['performance']['test_accuracy'],
                'cv_mean': training_record['performance']['cv_mean'],
                'improvement': improvement
            })
            
            self.version_counter += 1
            return True
            
        except Exception as e:
            self.log_retraining_event("MODEL_TRAINING", f"Model training failed: {str(e)}", False)
            return False
    
    def check_performance_improvement(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Check if current model performs better than previous"""
        if len(self.performance_tracking) < 2:
            return {'status': 'first_model', 'improvement': 0}
        
        # Get previous performance
        previous_performance = self.performance_tracking[-2]
        
        current_accuracy = current_performance['test_accuracy']
        previous_accuracy = previous_performance['test_accuracy']
        
        improvement = current_accuracy - previous_accuracy
        improvement_percent = (improvement / previous_accuracy) * 100 if previous_accuracy > 0 else 0
        
        return {
            'status': 'improved' if improvement > 0 else 'degraded',
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'current_accuracy': current_accuracy,
            'previous_accuracy': previous_accuracy
        }
    
    def run_complete_retraining_pipeline(self):
        """Run the complete retraining pipeline"""
        if self.is_running:
            self.log_retraining_event("RETRAINING", "Retraining already in progress, skipping", False)
            return
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            self.log_retraining_event("RETRAINING", "Starting automated retraining pipeline", True)
            
            # Step 1: Check and collect data
            if not self.collect_new_data():
                self.log_retraining_event("RETRAINING", "Failed to collect sufficient data", False)
                return
            
            # Step 2: Generate embeddings
            if not self.generate_embeddings():
                self.log_retraining_event("RETRAINING", "Failed to generate embeddings", False)
                return
            
            # Step 3: Train model
            embeddings_path = Path(config.EMBEDDINGS_DIR) / "latest_embeddings.npy"
            if not self.train_model(str(embeddings_path)):
                self.log_retraining_event("RETRAINING", "Failed to train model", False)
                return
            
            # Calculate total time
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.log_retraining_event("RETRAINING", f"Automated retraining pipeline completed successfully in {duration}", True, {
                'duration_minutes': duration.total_seconds() / 60,
                'version': f"1.{self.version_counter - 1}"
            })
            
        except Exception as e:
            self.log_retraining_event("RETRAINING", f"Retraining pipeline error: {str(e)}", False)
        finally:
            self.is_running = False
            self.save_training_history()
    
    def setup_schedule(self):
        """Setup the retraining schedule"""
        if config.TRAINING_FREQUENCY == "daily":
            schedule.every().day.at(config.TRAINING_TIME).do(self.run_complete_retraining_pipeline)
            logger.info(f"Scheduled daily retraining at {config.TRAINING_TIME}")
        elif config.TRAINING_FREQUENCY == "weekly":
            schedule.every().monday.at(config.TRAINING_TIME).do(self.run_complete_retraining_pipeline)
            logger.info(f"Scheduled weekly retraining on Mondays at {config.TRAINING_TIME}")
        else:
            logger.error(f"Unknown training frequency: {config.TRAINING_FREQUENCY}")
            return False
        
        return True
    
    def run_scheduler(self):
        """Run the automated retraining scheduler"""
        logger.info("Starting automated retraining scheduler")
        
        if not self.setup_schedule():
            return
        
        # Log next scheduled run
        next_run = schedule.next_run()
        if next_run:
            self.log_retraining_event("SCHEDULER", f"Next retraining scheduled for: {next_run}", True)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.log_retraining_event("SCHEDULER", "Scheduler stopped by user", True)
        except Exception as e:
            self.log_retraining_event("SCHEDULER", f"Scheduler error: {str(e)}", False)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_tracking:
            return {"message": "No performance data available"}
        
        # Calculate trends
        accuracies = [p['test_accuracy'] for p in self.performance_tracking]
        cv_means = [p['cv_mean'] for p in self.performance_tracking]
        
        return {
            'total_models': len(self.performance_tracking),
            'latest_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'avg_accuracy': np.mean(accuracies),
            'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
            'latest_cv_mean': cv_means[-1],
            'best_cv_mean': max(cv_means),
            'avg_cv_mean': np.mean(cv_means),
            'last_training': self.performance_tracking[-1]['timestamp']
        }
    
    def generate_performance_report(self) -> str:
        """Generate performance report"""
        summary = self.get_performance_summary()
        
        report = f"""
🤖 AUTOMATED RETRAINING SYSTEM - PERFORMANCE REPORT
{'='*60}

📊 Performance Summary:
  • Total Models Trained: {summary.get('total_models', 0)}
  • Latest Accuracy: {summary.get('latest_accuracy', 0):.4f}
  • Best Accuracy: {summary.get('best_accuracy', 0):.4f}
  • Average Accuracy: {summary.get('avg_accuracy', 0):.4f}
  • Accuracy Trend: {summary.get('accuracy_trend', 'unknown')}
  • Last Training: {summary.get('last_training', 'unknown')}

📈 Recent Performance History:
"""
        
        # Add recent performance data
        for i, perf in enumerate(self.performance_tracking[-5:]):
            report += f"  {i+1}. v{perf['version']}: {perf['test_accuracy']:.4f} ({perf['timestamp']})\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Automated Retraining System")
    parser.add_argument("--run-once", action="store_true", help="Run retraining pipeline once and exit")
    parser.add_argument("--frequency", type=str, choices=["daily", "weekly"], help="Override training frequency")
    parser.add_argument("--time", type=str, help="Override training time (HH:MM format)")
    parser.add_argument("--log-file", type=str, default="retraining_system.log", help="Log file path")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--data-count", type=int, help="Target number of images per class")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.frequency:
        config.TRAINING_FREQUENCY = args.frequency
    if args.time:
        config.TRAINING_TIME = args.time
    
    system = AutomatedRetrainingSystem(args.log_file)
    
    if args.report:
        print(system.generate_performance_report())
    elif args.run_once:
        print("🔄 Running retraining pipeline once...")
        system.run_complete_retraining_pipeline()
        print(system.generate_performance_report())
    else:
        print(f"🚀 Starting automated retraining system with {config.TRAINING_FREQUENCY} retraining at {config.TRAINING_TIME}")
        system.run_scheduler()

if __name__ == "__main__":
    main()
