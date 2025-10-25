"""
Training Report Viewer Utility
"""
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

def view_latest_report():
    """View the latest training report"""
    reports_dir = Path("training_reports")
    json_path = reports_dir / "latest_report.json"
    
    if not json_path.exists():
        print("❌ No training report found. Please run training first.")
        return
    
    with open(json_path, 'r') as f:
        report = json.load(f)
    
    print("📊 EXPOSR AUTHENTICITY CORE - TRAINING REPORT")
    print("=" * 60)
    print(f"🕒 Timestamp: {report['timestamp']}")
    print(f"🤖 Classifier: {report['model_info']['classifier_type']}")
    print(f"📏 Features: {report['model_info']['n_features']}")
    print(f"📊 Samples: {report['model_info']['n_samples']}")
    
    print(f"\n🎯 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"Overall Accuracy: {report['performance_metrics']['overall_accuracy']:.4f}")
    print(f"Training Accuracy: {report['performance_metrics']['training_accuracy']:.4f}")
    print(f"CV Mean: {report['performance_metrics']['cross_validation']['mean']:.4f}")
    print(f"CV Std: {report['performance_metrics']['cross_validation']['std']:.4f}")
    
    print(f"\n📈 PER-CLASS METRICS")
    print("-" * 30)
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    
    for class_name, metrics in report['detailed_metrics']['per_class'].items():
        print(f"{class_name:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print(f"\n🔍 CONFUSION MATRIX")
    print("-" * 30)
    cm = report['confusion_matrix']['matrix']
    print(f"{'':<10} {'Predicted AI':<15} {'Predicted Real':<15}")
    print(f"{'Actual AI':<10} {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"{'Actual Real':<10} {cm[1][0]:<15} {cm[1][1]:<15}")
    
    print(f"\n💡 INSIGHTS")
    print("-" * 30)
    print(f"True Positives: {report['confusion_matrix']['true_positives']}")
    print(f"True Negatives: {report['confusion_matrix']['true_negatives']}")
    print(f"False Positives: {report['confusion_matrix']['false_positives']}")
    print(f"False Negatives: {report['confusion_matrix']['false_negatives']}")

def list_reports():
    """List all available training reports"""
    reports_dir = Path("training_reports")
    
    if not reports_dir.exists():
        print("❌ No training reports directory found.")
        return
    
    json_files = list(reports_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No training reports found.")
        return
    
    print("📋 Available Training Reports:")
    print("-" * 40)
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                report = json.load(f)
            
            timestamp = datetime.fromisoformat(report['timestamp'].replace('Z', '+00:00'))
            print(f"📄 {json_file.name}")
            print(f"   🕒 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   🤖 {report['model_info']['classifier_type']}")
            print(f"   📊 Accuracy: {report['performance_metrics']['overall_accuracy']:.4f}")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")

def compare_reports():
    """Compare multiple training reports"""
    reports_dir = Path("training_reports")
    json_files = list(reports_dir.glob("*.json"))
    
    if len(json_files) < 2:
        print("❌ Need at least 2 reports to compare.")
        return
    
    print("📊 TRAINING REPORT COMPARISON")
    print("=" * 60)
    
    reports = []
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:  # Last 5 reports
        try:
            with open(json_file, 'r') as f:
                report = json.load(f)
            reports.append((json_file.name, report))
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")
    
    if not reports:
        print("❌ No valid reports found.")
        return
    
    print(f"{'Report':<20} {'Timestamp':<20} {'Accuracy':<10} {'CV Mean':<10}")
    print("-" * 60)
    
    for filename, report in reports:
        timestamp = datetime.fromisoformat(report['timestamp'].replace('Z', '+00:00'))
        print(f"{filename:<20} {timestamp.strftime('%m-%d %H:%M'):<20} "
              f"{report['performance_metrics']['overall_accuracy']:<10.4f} "
              f"{report['performance_metrics']['cross_validation']['mean']:<10.4f}")

def main():
    parser = argparse.ArgumentParser(description="Training Report Viewer")
    parser.add_argument("--view", action="store_true", help="View latest training report")
    parser.add_argument("--list", action="store_true", help="List all training reports")
    parser.add_argument("--compare", action="store_true", help="Compare multiple reports")
    
    args = parser.parse_args()
    
    if args.view:
        view_latest_report()
    elif args.list:
        list_reports()
    elif args.compare:
        compare_reports()
    else:
        # Default: view latest report
        view_latest_report()

if __name__ == "__main__":
    main()
