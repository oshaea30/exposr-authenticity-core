#!/bin/bash

# Exposr Authenticity Core - Setup and Training Script

echo "🚀 Setting up Exposr Authenticity Core..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{ai,real}
mkdir -p embeddings
mkdir -p models

# Set up environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "Please edit .env file with your API keys and preferences"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (optional)"
echo "2. Run: python scraper/scrape_ai.py --count 100"
echo "3. Run: python scraper/scrape_real.py --count 100"
echo "4. Run: python generate_embeddings.py"
echo "5. Run: python train.py"
echo "6. Run: python app.py"
echo ""
echo "Or run the scheduler: python scheduler.py"
