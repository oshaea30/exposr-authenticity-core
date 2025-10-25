"""
Utility functions for the Exposr Authenticity Core
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import config

logger = logging.getLogger(__name__)

def cleanup_old_data(keep_recent: int = 100):
    """Clean up old data files, keeping only the most recent ones"""
    logger.info(f"Cleaning up old data, keeping {keep_recent} most recent files")
    
    # Clean AI images
    ai_dir = Path(config.AI_IMAGES_DIR)
    if ai_dir.exists():
        ai_files = list(ai_dir.glob("*"))
        ai_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in ai_files[keep_recent:]:
            file.unlink()
            logger.debug(f"Deleted old AI image: {file}")
    
    # Clean real images
    real_dir = Path(config.REAL_IMAGES_DIR)
    if real_dir.exists():
        real_files = list(real_dir.glob("*"))
        real_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in real_files[keep_recent:]:
            file.unlink()
            logger.debug(f"Deleted old real image: {file}")
    
    logger.info("Data cleanup completed")

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_dataset_stats() -> Dict[str, Any]:
    """Get statistics about the dataset"""
    ai_dir = Path(config.AI_IMAGES_DIR)
    real_dir = Path(config.REAL_IMAGES_DIR)
    
    ai_count = len(list(ai_dir.glob("*"))) if ai_dir.exists() else 0
    real_count = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
    
    # Calculate total size
    ai_size = sum(f.stat().st_size for f in ai_dir.glob("*") if f.is_file()) if ai_dir.exists() else 0
    real_size = sum(f.stat().st_size for f in real_dir.glob("*") if f.is_file()) if real_dir.exists() else 0
    
    return {
        'ai_images': ai_count,
        'real_images': real_count,
        'total_images': ai_count + real_count,
        'ai_size_mb': ai_size / (1024 * 1024),
        'real_size_mb': real_size / (1024 * 1024),
        'total_size_mb': (ai_size + real_size) / (1024 * 1024)
    }

def backup_model(model_path: str, backup_dir: str = "backups"):
    """Create a backup of the trained model"""
    model_path = Path(model_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    # Create timestamped backup
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"classifier_{timestamp}.pkl"
    backup_path = backup_dir / backup_name
    
    # Copy model files
    shutil.copy2(model_path, backup_path)
    
    # Copy scaler if exists
    scaler_path = model_path.with_suffix('.scaler.pkl')
    if scaler_path.exists():
        scaler_backup = backup_dir / f"classifier_{timestamp}.scaler.pkl"
        shutil.copy2(scaler_path, scaler_backup)
    
    # Copy history if exists
    history_path = model_path.with_suffix('.history.pkl')
    if history_path.exists():
        history_backup = backup_dir / f"classifier_{timestamp}.history.pkl"
        shutil.copy2(history_path, history_backup)
    
    logger.info(f"Model backed up to {backup_path}")
    return backup_path

def restore_model(backup_path: str, model_path: str):
    """Restore model from backup"""
    backup_path = Path(backup_path)
    model_path = Path(model_path)
    
    if not backup_path.exists():
        logger.error(f"Backup file not found: {backup_path}")
        return False
    
    try:
        # Restore model
        shutil.copy2(backup_path, model_path)
        
        # Restore scaler if exists
        scaler_backup = backup_path.with_suffix('.scaler.pkl')
        if scaler_backup.exists():
            scaler_path = model_path.with_suffix('.scaler.pkl')
            shutil.copy2(scaler_backup, scaler_path)
        
        # Restore history if exists
        history_backup = backup_path.with_suffix('.history.pkl')
        if history_backup.exists():
            history_path = model_path.with_suffix('.history.pkl')
            shutil.copy2(history_backup, history_path)
        
        logger.info(f"Model restored from {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring model: {e}")
        return False

def list_backups(backup_dir: str = "backups") -> List[Dict[str, Any]]:
    """List available model backups"""
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        return []
    
    backups = []
    for file in backup_dir.glob("classifier_*.pkl"):
        if not file.name.endswith('.scaler.pkl') and not file.name.endswith('.history.pkl'):
            stat = file.stat()
            backups.append({
                'filename': file.name,
                'path': str(file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            })
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x['created'], reverse=True)
    
    return backups

def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements"""
    import torch
    import transformers
    import sklearn
    
    requirements = {
        'pytorch_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'sklearn_version': sklearn.__version__,
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    return requirements
