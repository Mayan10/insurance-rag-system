#!/bin/bash

echo "🚀 Setting up ngrok for Insurance RAG System"

# Check if ngrok is already installed
if command -v ngrok &> /dev/null; then
    echo "✅ ngrok is already installed"
    ngrok version
else
    echo "📥 Installing ngrok..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "🍺 Installing via Homebrew..."
            brew install ngrok
        else
            echo "📥 Downloading ngrok for macOS..."
            curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip -o ngrok.zip
            unzip ngrok.zip
            sudo mv ngrok /usr/local/bin/
            rm ngrok.zip
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "📥 Downloading ngrok for Linux..."
        curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip -o ngrok.zip
        unzip ngrok.zip
        sudo mv ngrok /usr/local/bin/
        rm ngrok.zip
    else
        echo "❌ Unsupported OS. Please install ngrok manually from https://ngrok.com"
        exit 1
    fi
fi

echo ""
echo "🔐 Next steps:"
echo "1. Sign up at https://ngrok.com (free)"
echo "2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken"
echo "3. Run: ngrok config add-authtoken YOUR_AUTHTOKEN_HERE"
echo ""
echo "🚀 Then start your API:"
echo "1. python app.py"
echo "2. In another terminal: ngrok http 8000"
echo "3. Copy the HTTPS URL and update test_ngrok.py"
echo ""
echo "✅ Setup complete!" 