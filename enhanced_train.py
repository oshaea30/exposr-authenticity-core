"""
Enhanced Classifier Training with Performance Tracking
"""
import os
import numpy as np
import pickle
import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import config
from embed_images import EmbeddingPipeline

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EnhancedClassifierTrainer:
    def __init__(self, classifier_type: str = config.CLASSIFIER_TYPE, version: str = "1.0"):
        self.classifier_type = classifier_type
        self.version = version
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        self.reports_dir = Path("training_reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.load_performance_history()
        
    def load_performance_history(self):
        """Load historical performance data"""
        history_file = self.reports_dir / "performance_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
        else:
            self.performance_history = []
    
    def save_performance_history(self):
        """Save performance history"""
        history_file = self.reports_dir / "performance_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def load_data_with_validation(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load embeddings with comprehensive validation"""
        embeddings_path = Path(embeddings_path)
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        metadata_path = embeddings_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Validate embeddings
        if len(embeddings) == 0:
            raise ValueError("No embeddings found")
        
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain invalid values")
        
        # Create labels
        ai_count = metadata.get('ai_count', len(embeddings) // 2)
        real_count = metadata.get('real_count', len(embeddings) // 2)
        
        labels = np.array([0] * ai_count + [1] * real_count)
        
        # Validate label count matches embedding count
        if len(labels) != len(embeddings):
            logger.warning(f"Label count ({len(labels)}) doesn't match embedding count ({len(embeddings)})")
            # Adjust labels to match embeddings
            if len(embeddings) > len(labels):
                # Pad with zeros (assume AI)
                labels = np.pad(labels, (0, len(embeddings) - len(labels)), 'constant', constant_values=0)
            else:
                # Truncate labels
                labels = labels[:len(embeddings)]
        
        logger.info(f"Loaded {len(embeddings)} embeddings: {ai_count} AI, {real_count} real")
        
        return embeddings, labels, metadata
    
    def create_classifier(self, hyperparams: Dict[str, Any] = None) -> Any:
        """Create classifier with optional hyperparameters"""
        params = hyperparams or {}
        
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.classifier_type == "logistic_regression":
            return LogisticRegression(
                C=params.get('C', 1.0),
                penalty=params.get('penalty', 'l2'),
                solver=params.get('solver', 'liblinear'),
                random_state=config.RANDOM_STATE,
                max_iter=params.get('max_iter', 1000)
            )
        elif self.classifier_type == "svm":
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
                random_state=config.RANDOM_STATE,
                probability=True
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Enhanced hyperparameter optimization with better search space"""
        logger.info("Starting hyperparameter optimization...")
        
        # Define search spaces based on classifier type
        if self.classifier_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif self.classifier_type == "logistic_regression":
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000]
            }
        elif self.classifier_type == "svm":
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'kernel': ['rbf', 'linear', 'poly']
            }
        else:
            logger.warning(f"No hyperparameter optimization for {self.classifier_type}")
            return {}
        
        # Use stratified k-fold for better validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        
        grid_search = GridSearchCV(
            self.create_classifier(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_with_validation(self, embeddings_path: str, optimize: bool = True) -> Dict[str, Any]:
        """Enhanced training with comprehensive validation"""
        logger.info("Starting enhanced classifier training")
        
        # Load and validate data
        X, y, metadata = self.load_data_with_validation(embeddings_path)
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter optimization
        best_params = {}
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
        
        # Create and train classifier
        self.model = self.create_classifier(best_params)
        
        logger.info("Training classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Comprehensive evaluation
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation with stratified k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv)
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate additional metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # ROC AUC if probabilities available
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                logger.info(f"ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Classification report
        class_names = ['AI', 'Real']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print detailed performance metrics
        self.print_enhanced_metrics(y_test, y_pred, class_names, roc_auc)
        
        # Create comprehensive training record
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'version': self.version,
            'classifier_type': self.classifier_type,
            'best_params': best_params,
            'data_metadata': metadata,
            'performance': {
                'train_accuracy': float(train_score),
                'test_accuracy': float(test_score),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'roc_auc': float(roc_auc) if roc_auc else None,
                'precision_per_class': precision.tolist(),
                'recall_per_class': recall.tolist(),
                'f1_per_class': f1.tolist(),
                'support_per_class': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Save reports
        self.save_enhanced_reports(training_record)
        
        # Update performance history
        self.performance_history.append(training_record)
        self.save_performance_history()
        
        # Store training history
        self.training_history = training_record
        
        # Create visualizations
        self.create_enhanced_plots(cm, class_names, y_test, y_pred_proba)
        
        return training_record
    
    def print_enhanced_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, class_names: list, roc_auc: float = None):
        """Print enhanced performance metrics"""
        print("\n" + "="*70)
        print("🎯 ENHANCED TRAINING PERFORMANCE METRICS")
        print("="*70)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if roc_auc:
            print(f"📈 ROC AUC Score: {roc_auc:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        print(f"\n📈 Per-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        print("-" * 60)
        print(f"{'Macro Avg':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        print(f"{'Weighted':<10} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print("-" * 40)
        print(f"{'':<12} {'Predicted':<25}")
        print(f"{'Actual':<12} {'AI':<12} {'Real':<12}")
        print("-" * 40)
        print(f"{'AI':<12} {cm[0,0]:<12} {cm[0,1]:<12}")
        print(f"{'Real':<12} {cm[1,0]:<12} {cm[1,1]:<12}")
        print("-" * 40)
        
        # Additional insights
        print(f"\n💡 Additional Insights:")
        print(f"   • True Positives (Real detected as Real): {cm[1,1]}")
        print(f"   • True Negatives (AI detected as AI): {cm[0,0]}")
        print(f"   • False Positives (AI detected as Real): {cm[0,1]}")
        print(f"   • False Negatives (Real detected as AI): {cm[1,0]}")
        
        if cm[0,1] > 0 or cm[1,0] > 0:
            print(f"   • False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1]):.4f}")
            print(f"   • False Negative Rate: {cm[1,0]/(cm[1,0]+cm[1,1]):.4f}")
        
        print("="*70)
    
    def save_enhanced_reports(self, training_record: Dict[str, Any]):
        """Save enhanced reports in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = self.reports_dir / f"training_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(training_record, f, indent=2)
        
        # Latest report
        latest_json_path = self.reports_dir / "latest_report.json"
        with open(latest_json_path, 'w') as f:
            json.dump(training_record, f, indent=2)
        
        # CSV report
        csv_path = self.reports_dir / f"training_report_{timestamp}.csv"
        self.save_csv_report(training_record, csv_path)
        
        # TXT report
        txt_path = self.reports_dir / f"training_report_{timestamp}.txt"
        self.save_txt_report(training_record, txt_path)
        
        logger.info(f"Enhanced reports saved:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  TXT: {txt_path}")
    
    def save_csv_report(self, training_record: Dict[str, Any], csv_path: Path):
        """Save enhanced CSV report"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Metric", "Value"])
            writer.writerow([])
            
            # Basic info
            writer.writerow(["TIMESTAMP", training_record["timestamp"]])
            writer.writerow(["VERSION", training_record["version"]])
            writer.writerow(["CLASSIFIER_TYPE", training_record["classifier_type"]])
            writer.writerow([])
            
            # Performance metrics
            perf = training_record["performance"]
            writer.writerow(["PERFORMANCE METRICS"])
            writer.writerow(["Training Accuracy", f"{perf['train_accuracy']:.4f}"])
            writer.writerow(["Test Accuracy", f"{perf['test_accuracy']:.4f}"])
            writer.writerow(["CV Mean", f"{perf['cv_mean']:.4f}"])
            writer.writerow(["CV Std", f"{perf['cv_std']:.4f}"])
            if perf['roc_auc']:
                writer.writerow(["ROC AUC", f"{perf['roc_auc']:.4f}"])
            writer.writerow([])
            
            # Per-class metrics
            writer.writerow(["PER-CLASS METRICS"])
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            
            class_names = ['AI', 'Real']
            for i, class_name in enumerate(class_names):
                writer.writerow([
                    class_name,
                    f"{perf['precision_per_class'][i]:.4f}",
                    f"{perf['recall_per_class'][i]:.4f}",
                    f"{perf['f1_per_class'][i]:.4f}",
                    perf['support_per_class'][i]
                ])
    
    def save_txt_report(self, training_record: Dict[str, Any], txt_path: Path):
        """Save enhanced TXT report"""
        with open(txt_path, 'w') as f:
            f.write("EXPOSR AUTHENTICITY CORE - ENHANCED TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {training_record['timestamp']}\n")
            f.write(f"Version: {training_record['version']}\n")
            f.write(f"Classifier Type: {training_record['classifier_type']}\n")
            f.write(f"Best Parameters: {training_record['best_params']}\n\n")
            
            perf = training_record["performance"]
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Training Accuracy: {perf['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {perf['test_accuracy']:.4f}\n")
            f.write(f"CV Mean: {perf['cv_mean']:.4f}\n")
            f.write(f"CV Std: {perf['cv_std']:.4f}\n")
            if perf['roc_auc']:
                f.write(f"ROC AUC: {perf['roc_auc']:.4f}\n")
            f.write("\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            
            class_names = ['AI', 'Real']
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<10} {perf['precision_per_class'][i]:<12.4f} "
                       f"{perf['recall_per_class'][i]:<12.4f} {perf['f1_per_class'][i]:<12.4f} "
                       f"{perf['support_per_class'][i]:<10}\n")
    
    def create_enhanced_plots(self, cm: np.ndarray, class_names: list, y_test: np.ndarray, y_pred_proba: np.ndarray = None):
        """Create enhanced visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_features = np.argsort(importances)[-20:]  # Top 20 features
            
            axes[0,1].barh(range(len(top_features)), importances[top_features])
            axes[0,1].set_title('Top 20 Feature Importances')
            axes[0,1].set_xlabel('Importance')
            axes[0,1].set_ylabel('Feature Index')
        else:
            axes[0,1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this classifier', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Feature Importance')
        
        # ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1,0].set_xlim([0.0, 1.0])
            axes[1,0].set_ylim([0.0, 1.05])
            axes[1,0].set_xlabel('False Positive Rate')
            axes[1,0].set_ylabel('True Positive Rate')
            axes[1,0].set_title('ROC Curve')
            axes[1,0].legend(loc="lower right")
        else:
            axes[1,0].text(0.5, 0.5, 'ROC curve\nnot available\n(no probabilities)', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('ROC Curve')
        
        # Performance History
        if len(self.performance_history) > 1:
            timestamps = [record['timestamp'] for record in self.performance_history[-10:]]
            accuracies = [record['performance']['test_accuracy'] for record in self.performance_history[-10:]]
            
            axes[1,1].plot(range(len(accuracies)), accuracies, marker='o')
            axes[1,1].set_title('Performance Over Time')
            axes[1,1].set_xlabel('Training Run')
            axes[1,1].set_ylabel('Test Accuracy')
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Performance history\nwill appear after\nmultiple training runs', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Performance Over Time')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.models_dir / f"enhanced_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced training plots saved to {plot_path}")
    
    def save_model_with_version(self, model_path: str):
        """Save model with version tracking"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create versioned filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = model_path.parent / f"classifier_v{self.version}_{timestamp}.pkl"
        
        # Save model
        with open(versioned_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = versioned_path.with_suffix('.scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        history_path = versioned_path.with_suffix('.history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Also save as latest
        latest_path = model_path
        latest_scaler_path = model_path.with_suffix('.scaler.pkl')
        latest_history_path = model_path.with_suffix('.history.json')
        
        with open(latest_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(latest_scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(latest_history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {versioned_path}")
        logger.info(f"Latest model saved to {latest_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Classifier Training")
    parser.add_argument("--embeddings", type=str, default="embeddings/latest_embeddings.npy", help="Path to embeddings file")
    parser.add_argument("--output", type=str, default="models/classifier.pkl", help="Output path for trained model")
    parser.add_argument("--classifier", type=str, default=config.CLASSIFIER_TYPE, choices=["random_forest", "logistic_regression", "svm"], help="Classifier type")
    parser.add_argument("--version", type=str, default="1.0", help="Model version")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Check if embeddings exist
    if not os.path.exists(args.embeddings):
        print(f"Embeddings file not found: {args.embeddings}")
        print("Please run embed_images.py first")
        return
    
    # Initialize trainer
    trainer = EnhancedClassifierTrainer(args.classifier, args.version)
    
    # Train model
    training_record = trainer.train_with_validation(args.embeddings, optimize=not args.no_optimize)
    
    # Save model
    trainer.save_model_with_version(args.output)
    
    # Print results
    print(f"\n🎉 Enhanced training completed!")
    print(f"📊 Test accuracy: {training_record['performance']['test_accuracy']:.4f}")
    print(f"📈 Cross-validation mean: {training_record['performance']['cv_mean']:.4f} (+/- {training_record['performance']['cv_std'] * 2:.4f})")
    if training_record['performance']['roc_auc']:
        print(f"📊 ROC AUC: {training_record['performance']['roc_auc']:.4f}")
    print(f"💾 Model saved to {args.output}")
    print(f"📋 Enhanced reports saved to training_reports/")

if __name__ == "__main__":
    main()
