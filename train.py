"""
Classifier Training Pipeline
"""
import os
import numpy as np
import pickle
import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import config
from generate_embeddings import CLIPEmbeddingGenerator

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ClassifierTrainer:
    def __init__(self, classifier_type: str = config.CLASSIFIER_TYPE):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        self.reports_dir = Path("training_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_data(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings and create labels"""
        embeddings_path = Path(embeddings_path)
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        metadata_path = embeddings_path.with_suffix('.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create labels (0 for AI, 1 for Real)
        ai_count = metadata['ai_count']
        real_count = metadata['real_count']
        
        labels = np.array([0] * ai_count + [1] * real_count)
        
        logger.info(f"Loaded {len(embeddings)} embeddings: {ai_count} AI, {real_count} real")
        
        return embeddings, labels
    
    def create_classifier(self) -> Any:
        """Create classifier based on type"""
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.classifier_type == "logistic_regression":
            return LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
                C=1.0
            )
        elif self.classifier_type == "svm":
            return SVC(
                kernel='rbf',
                random_state=config.RANDOM_STATE,
                probability=True,
                C=1.0,
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV"""
        logger.info("Optimizing hyperparameters...")
        
        if self.classifier_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.classifier_type == "logistic_regression":
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.classifier_type == "svm":
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        else:
            logger.warning(f"No hyperparameter optimization for {self.classifier_type}")
            return {}
        
        # Use smaller grid for faster training
        if len(X_train) > 1000:
            # Reduce grid size for large datasets
            if self.classifier_type == "random_forest":
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, None],
                    'min_samples_split': [2, 5]
                }
            elif self.classifier_type == "logistic_regression":
                param_grid = {
                    'C': [1.0, 10.0],
                    'penalty': ['l2']
                }
            elif self.classifier_type == "svm":
                param_grid = {
                    'C': [1.0, 10.0],
                    'gamma': ['scale', 'auto']
                }
        
        grid_search = GridSearchCV(
            self.create_classifier(),
            param_grid,
            cv=3,  # Reduced CV folds for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def print_performance_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, class_names: list):
        """Print detailed performance metrics to console"""
        print("\n" + "="*60)
        print("🎯 TRAINING PERFORMANCE METRICS")
        print("="*60)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Precision, Recall, F1-score per class
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        print(f"\n📈 Per-Class Metrics:")
        print("-" * 50)
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 50)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        print("-" * 50)
        print(f"{'Macro Avg':<10} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        print(f"{'Weighted':<10} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print("-" * 30)
        print(f"{'':<8} {'Predicted':<20}")
        print(f"{'Actual':<8} {'AI':<10} {'Real':<10}")
        print("-" * 30)
        print(f"{'AI':<8} {cm[0,0]:<10} {cm[0,1]:<10}")
        print(f"{'Real':<8} {cm[1,0]:<10} {cm[1,1]:<10}")
        print("-" * 30)
        
        # Additional insights
        print(f"\n💡 Additional Insights:")
        print(f"   • True Positives (Real detected as Real): {cm[1,1]}")
        print(f"   • True Negatives (AI detected as AI): {cm[0,0]}")
        print(f"   • False Positives (AI detected as Real): {cm[0,1]}")
        print(f"   • False Negatives (Real detected as AI): {cm[1,0]}")
        
        if cm[0,1] > 0 or cm[1,0] > 0:
            print(f"   • False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1]):.4f}")
            print(f"   • False Negative Rate: {cm[1,0]/(cm[1,0]+cm[1,1]):.4f}")
        
        print("="*60)
    
    def save_json_report(self, y_test: np.ndarray, y_pred: np.ndarray, class_names: list, 
                        train_score: float, test_score: float, cv_scores: np.ndarray, 
                        best_params: Dict[str, Any] = None):
        """Save comprehensive training report as JSON"""
        timestamp = datetime.now().isoformat()
        
        # Calculate detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report as dictionary
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Create comprehensive report
        report = {
            "timestamp": timestamp,
            "model_info": {
                "classifier_type": self.classifier_type,
                "best_hyperparameters": best_params or {},
                "n_features": len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else 0,
                "n_samples": len(y_test) + len(y_pred)  # Approximate total samples
            },
            "performance_metrics": {
                "overall_accuracy": float(test_score),
                "training_accuracy": float(train_score),
                "cross_validation": {
                    "scores": cv_scores.tolist(),
                    "mean": float(cv_scores.mean()),
                    "std": float(cv_scores.std()),
                    "min": float(cv_scores.min()),
                    "max": float(cv_scores.max())
                }
            },
            "detailed_metrics": {
                "per_class": {
                    class_names[i]: {
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1_score": float(f1[i]),
                        "support": int(support[i])
                    } for i in range(len(class_names))
                },
                "macro_averages": {
                    "precision": float(np.mean(precision)),
                    "recall": float(np.mean(recall)),
                    "f1_score": float(np.mean(f1))
                },
                "weighted_averages": {
                    "precision": float(precision_recall_fscore_support(y_test, y_pred, average='weighted')[0]),
                    "recall": float(precision_recall_fscore_support(y_test, y_pred, average='weighted')[1]),
                    "f1_score": float(precision_recall_fscore_support(y_test, y_pred, average='weighted')[2])
                }
            },
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": class_names,
                "true_positives": int(cm[1,1]),
                "true_negatives": int(cm[0,0]),
                "false_positives": int(cm[0,1]),
                "false_negatives": int(cm[1,0])
            },
            "classification_report": class_report
        }
        
        # Save JSON report
        json_path = self.reports_dir / "latest_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"JSON training report saved to {json_path}")
        return report
    
    def save_csv_report(self, report: Dict[str, Any]):
        """Save training report as CSV for visual inspection"""
        csv_path = self.reports_dir / "latest_report.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Metric", "Value"])
            writer.writerow([])
            
            # Basic info
            writer.writerow(["TIMESTAMP", report["timestamp"]])
            writer.writerow(["CLASSIFIER_TYPE", report["model_info"]["classifier_type"]])
            writer.writerow([])
            
            # Performance metrics
            writer.writerow(["PERFORMANCE METRICS"])
            writer.writerow(["Overall Accuracy", f"{report['performance_metrics']['overall_accuracy']:.4f}"])
            writer.writerow(["Training Accuracy", f"{report['performance_metrics']['training_accuracy']:.4f}"])
            writer.writerow(["CV Mean", f"{report['performance_metrics']['cross_validation']['mean']:.4f}"])
            writer.writerow(["CV Std", f"{report['performance_metrics']['cross_validation']['std']:.4f}"])
            writer.writerow([])
            
            # Per-class metrics
            writer.writerow(["PER-CLASS METRICS"])
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            
            for class_name, metrics in report["detailed_metrics"]["per_class"].items():
                writer.writerow([
                    class_name,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1_score']:.4f}",
                    metrics['support']
                ])
            
            writer.writerow([])
            
            # Macro averages
            writer.writerow(["MACRO AVERAGES"])
            writer.writerow(["Precision", f"{report['detailed_metrics']['macro_averages']['precision']:.4f}"])
            writer.writerow(["Recall", f"{report['detailed_metrics']['macro_averages']['recall']:.4f}"])
            writer.writerow(["F1-Score", f"{report['detailed_metrics']['macro_averages']['f1_score']:.4f}"])
            writer.writerow([])
            
            # Confusion matrix
            writer.writerow(["CONFUSION MATRIX"])
            writer.writerow(["", "Predicted AI", "Predicted Real"])
            writer.writerow(["Actual AI", report['confusion_matrix']['matrix'][0][0], report['confusion_matrix']['matrix'][0][1]])
            writer.writerow(["Actual Real", report['confusion_matrix']['matrix'][1][0], report['confusion_matrix']['matrix'][1][1]])
        
        logger.info(f"CSV training report saved to {csv_path}")
    
    def save_txt_report(self, report: Dict[str, Any]):
        """Save training report as TXT for easy reading"""
        txt_path = self.reports_dir / "latest_report.txt"
        
        with open(txt_path, 'w') as f:
            f.write("EXPOSR AUTHENTICITY CORE - TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Classifier Type: {report['model_info']['classifier_type']}\n")
            f.write(f"Number of Features: {report['model_info']['n_features']}\n")
            f.write(f"Number of Samples: {report['model_info']['n_samples']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Accuracy: {report['performance_metrics']['overall_accuracy']:.4f}\n")
            f.write(f"Training Accuracy: {report['performance_metrics']['training_accuracy']:.4f}\n")
            f.write(f"Cross-Validation Mean: {report['performance_metrics']['cross_validation']['mean']:.4f}\n")
            f.write(f"Cross-Validation Std: {report['performance_metrics']['cross_validation']['std']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 50 + "\n")
            
            for class_name, metrics in report["detailed_metrics"]["per_class"].items():
                f.write(f"{class_name:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}\n")
            
            f.write("\nMACRO AVERAGES\n")
            f.write("-" * 15 + "\n")
            f.write(f"Precision: {report['detailed_metrics']['macro_averages']['precision']:.4f}\n")
            f.write(f"Recall: {report['detailed_metrics']['macro_averages']['recall']:.4f}\n")
            f.write(f"F1-Score: {report['detailed_metrics']['macro_averages']['f1_score']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 15 + "\n")
            f.write(f"{'':<10} {'Predicted AI':<15} {'Predicted Real':<15}\n")
            f.write(f"{'Actual AI':<10} {report['confusion_matrix']['matrix'][0][0]:<15} {report['confusion_matrix']['matrix'][0][1]:<15}\n")
            f.write(f"{'Actual Real':<10} {report['confusion_matrix']['matrix'][1][0]:<15} {report['confusion_matrix']['matrix'][1][1]:<15}\n\n")
            
            f.write("ADDITIONAL INSIGHTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"True Positives (Real detected as Real): {report['confusion_matrix']['true_positives']}\n")
            f.write(f"True Negatives (AI detected as AI): {report['confusion_matrix']['true_negatives']}\n")
            f.write(f"False Positives (AI detected as Real): {report['confusion_matrix']['false_positives']}\n")
            f.write(f"False Negatives (Real detected as AI): {report['confusion_matrix']['false_negatives']}\n")
        
        logger.info(f"TXT training report saved to {txt_path}")
    
    def train(self, embeddings_path: str, optimize: bool = True) -> Dict[str, Any]:
        """Train the classifier"""
        logger.info("Starting classifier training")
        
        # Load data
        X, y = self.load_data(embeddings_path)
        
        if len(X) == 0:
            raise ValueError("No embeddings found. Please generate embeddings first.")
        
        # Split data
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
        
        # Create and train classifier
        self.model = self.create_classifier()
        
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
            if best_params:
                # Update model with best parameters
                if self.classifier_type == "random_forest":
                    self.model = RandomForestClassifier(**best_params, random_state=config.RANDOM_STATE, n_jobs=-1)
                elif self.classifier_type == "logistic_regression":
                    self.model = LogisticRegression(**best_params, random_state=config.RANDOM_STATE, max_iter=1000)
                elif self.classifier_type == "svm":
                    self.model = SVC(**best_params, random_state=config.RANDOM_STATE, probability=True)
        
        # Train model
        logger.info("Training classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Classification report
        class_names = ['AI', 'Real']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print detailed performance metrics to console
        self.print_performance_metrics(y_test, y_pred, class_names)
        
        # Save comprehensive reports
        best_params = self.optimize_hyperparameters(X_train_scaled, y_train) if optimize else {}
        json_report = self.save_json_report(y_test, y_pred, class_names, train_score, test_score, cv_scores, best_params)
        self.save_csv_report(json_report)
        self.save_txt_report(json_report)
        
        # Store training history
        self.training_history = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'classifier_type': self.classifier_type,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'best_params': best_params
        }
        
        # Plot results
        self.plot_results(cm, class_names)
        
        return self.training_history
    
    def plot_results(self, cm: np.ndarray, class_names: list):
        """Plot confusion matrix and feature importance"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_features = np.argsort(importances)[-20:]  # Top 20 features
            
            axes[1].barh(range(len(top_features)), importances[top_features])
            axes[1].set_title('Top 20 Feature Importances')
            axes[1].set_xlabel('Importance')
            axes[1].set_ylabel('Feature Index')
        else:
            axes[1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this classifier', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(config.MODELS_DIR) / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
    
    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = model_path.with_suffix('.scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        history_path = model_path.with_suffix('.history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Training history saved to {history_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        model_path = Path(model_path)
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = model_path.with_suffix('.scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train classifier for AI/Real image detection")
    parser.add_argument("--embeddings", type=str, default=os.path.join(config.EMBEDDINGS_DIR, "image_vectors.npy"), help="Path to embeddings file")
    parser.add_argument("--output", type=str, default=os.path.join(config.MODELS_DIR, "classifier.pkl"), help="Output path for trained model")
    parser.add_argument("--classifier", type=str, default=config.CLASSIFIER_TYPE, choices=["random_forest", "logistic_regression", "svm"], help="Classifier type")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Check if embeddings exist
    if not os.path.exists(args.embeddings):
        print(f"Embeddings file not found: {args.embeddings}")
        print("Please run generate_embeddings.py first")
        return
    
    # Initialize trainer
    trainer = ClassifierTrainer(args.classifier)
    
    # Train model
    history = trainer.train(args.embeddings, optimize=not args.no_optimize)
    
    # Save model
    trainer.save_model(args.output)
    
    # Print results
    print(f"\n🎉 Training completed!")
    print(f"📊 Test accuracy: {history['test_accuracy']:.4f}")
    print(f"📈 Cross-validation mean: {history['cv_mean']:.4f} (+/- {history['cv_std'] * 2:.4f})")
    print(f"💾 Model saved to {args.output}")
    print(f"📋 Training reports saved to training_reports/")
    print(f"   • latest_report.json - Complete metrics in JSON format")
    print(f"   • latest_report.csv - Metrics in CSV format for analysis")
    print(f"   • latest_report.txt - Human-readable report")

if __name__ == "__main__":
    main()
