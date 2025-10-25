"""
Test script to demonstrate enhanced training reporting
"""
import os
import sys
import numpy as np
from pathlib import Path
import config

def create_sample_data():
    """Create sample embeddings and labels for testing"""
    print("🧪 Creating sample data for testing...")
    
    # Create sample embeddings (512-dimensional CLIP embeddings)
    n_samples = 100
    n_features = 512
    
    # Generate random embeddings
    np.random.seed(42)
    ai_embeddings = np.random.randn(50, n_features)
    real_embeddings = np.random.randn(50, n_features)
    
    # Combine embeddings
    all_embeddings = np.vstack([ai_embeddings, real_embeddings])
    
    # Create labels (0 for AI, 1 for Real)
    labels = np.array([0] * 50 + [1] * 50)
    
    # Shuffle data
    indices = np.random.permutation(len(all_embeddings))
    all_embeddings = all_embeddings[indices]
    labels = labels[indices]
    
    # Save embeddings
    embeddings_path = Path(config.EMBEDDINGS_DIR) / "image_vectors.npy"
    embeddings_path.parent.mkdir(exist_ok=True)
    np.save(embeddings_path, all_embeddings)
    
    # Save metadata
    metadata_path = embeddings_path.with_suffix('.pkl')
    import pickle
    metadata = {
        'ai_count': 50,
        'real_count': 50,
        'total_count': 100,
        'embedding_dimension': n_features,
        'labels': ['AI'] * 50 + ['Real'] * 50,
        'model_name': config.CLIP_MODEL_NAME
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Sample data created: {embeddings_path}")
    return embeddings_path

def test_training_reporting():
    """Test the enhanced training reporting functionality"""
    print("🚀 Testing Enhanced Training Reporting...")
    
    # Create sample data
    embeddings_path = create_sample_data()
    
    # Import trainer
    from train import ClassifierTrainer
    
    # Initialize trainer
    trainer = ClassifierTrainer("random_forest")
    
    print("\n📊 Running training with enhanced reporting...")
    
    # Train model (this will generate all reports)
    history = trainer.train(str(embeddings_path), optimize=False)  # Skip optimization for speed
    
    # Save model
    model_path = Path(config.MODELS_DIR) / "test_classifier.pkl"
    trainer.save_model(str(model_path))
    
    print("\n📋 Checking generated reports...")
    
    # Check if reports were created
    reports_dir = Path("training_reports")
    
    json_report = reports_dir / "latest_report.json"
    csv_report = reports_dir / "latest_report.csv"
    txt_report = reports_dir / "latest_report.txt"
    
    if json_report.exists():
        print(f"✅ JSON report created: {json_report}")
    else:
        print(f"❌ JSON report not found: {json_report}")
    
    if csv_report.exists():
        print(f"✅ CSV report created: {csv_report}")
    else:
        print(f"❌ CSV report not found: {csv_report}")
    
    if txt_report.exists():
        print(f"✅ TXT report created: {txt_report}")
    else:
        print(f"❌ TXT report not found: {txt_report}")
    
    # Test report viewer
    print("\n🔍 Testing report viewer...")
    try:
        from view_reports import view_latest_report
        view_latest_report()
    except Exception as e:
        print(f"❌ Error testing report viewer: {e}")
    
    print("\n🎉 Enhanced training reporting test completed!")
    print("📁 Check the training_reports/ directory for generated reports")

def main():
    """Main test function"""
    print("🧪 EXPOSR AUTHENTICITY CORE - ENHANCED TRAINING REPORTING TEST")
    print("=" * 70)
    
    try:
        test_training_reporting()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    main()
