# Exposr Authenticity Core

A machine learning pipeline for detecting AI-generated images using CLIP embeddings and scikit-learn classifiers.

## Overview

This system scrapes labeled datasets of AI-generated (Midjourney) and real images (Wikimedia/Unsplash), converts them to vector embeddings using OpenAI CLIP, and trains a classifier to distinguish between them.

## Features

- **Automated Data Collection**: Scrapers for AI and real image datasets
- **CLIP Embeddings**: State-of-the-art image vectorization using OpenAI CLIP
- **Flexible Training**: Support for multiple classifier algorithms
- **Comprehensive Reporting**: Detailed training metrics with JSON, CSV, and TXT reports
- **Scheduled Retraining**: Configurable daily/weekly model updates
- **REST API**: Ready for integration with Exposr platform
- **Scalable Architecture**: Modular design for easy extension

## Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

3. **Scrape Training Data**:

   ```bash
   python scraper/scrape_ai.py --count 1000
   python scraper/scrape_real.py --count 1000
   ```

4. **Generate Embeddings**:

   ```bash
   python generate_embeddings.py
   ```

5. **Train Model**:

   ```bash
   python train.py
   ```

6. **View Training Reports**:

   ```bash
   python view_reports.py --view    # View latest report
   python view_reports.py --list    # List all reports
   python view_reports.py --compare # Compare multiple reports
   ```

7. **Classify New Images**:

   ```bash
   python classify.py --image path/to/image.jpg
   ```

8. **Start API Server**:
   ```bash
   python app.py
   ```

## Project Structure

```
exposr-authenticity-core/
├── data/
│   ├── ai/               # Scraped AI images
│   └── real/             # Scraped real images
├── embeddings/
│   └── image_vectors.npy # CLIP embeddings output
├── models/
│   └── classifier.pkl    # Trained scikit-learn model
├── scraper/
│   ├── scrape_ai.py      # Midjourney image scraper
│   └── scrape_real.py    # Wikimedia/Unsplash scraper
├── train.py              # Model training pipeline
├── classify.py           # Inference pipeline
├── generate_embeddings.py # CLIP embedding generation
├── scheduler.py          # Training scheduler
├── app.py                # Flask REST API
├── config.py             # Configuration management
├── view_reports.py        # Training report viewer
└── README.md
```

## Training Reports

The system automatically generates comprehensive training reports in multiple formats:

### Console Output

During training, you'll see detailed performance metrics including:

- Overall accuracy
- Precision, recall, and F1-score per class
- Confusion matrix
- Additional insights (false positive/negative rates)

### Report Files

After each training session, reports are saved to `training_reports/`:

- **`latest_report.json`**: Complete metrics in JSON format for programmatic access
- **`latest_report.csv`**: Metrics in CSV format for spreadsheet analysis
- **`latest_report.txt`**: Human-readable report for quick review

### Report Contents

Each report includes:

- Training timestamp and model information
- Overall and per-class performance metrics
- Cross-validation scores
- Confusion matrix with detailed breakdown
- Macro and weighted averages
- Hyperparameter optimization results

### Viewing Reports

```bash
# View latest report
python view_reports.py --view

# List all available reports
python view_reports.py --list

# Compare multiple reports
python view_reports.py --compare
```

## Configuration

Set training frequency in `config.py`:

- `TRAINING_FREQUENCY = "daily"` or `"weekly"`
- `TRAINING_TIME = "02:00"` (24-hour format)

## API Endpoints

- `POST /classify` - Classify a single image
- `POST /batch_classify` - Classify multiple images
- `GET /model_info` - Get model performance metrics
- `POST /retrain` - Trigger manual retraining

## Performance

- **Accuracy**: ~85-90% on test datasets
- **Speed**: ~100ms per image classification
- **Scalability**: Handles batch processing efficiently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
