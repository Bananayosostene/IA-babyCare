#!/bin/bash

# Setup script for Django WebSocket baby monitor

echo "🚀 Setting up Django WebSocket Baby Monitor..."

# Install Redis (required for Django Channels)
echo "📦 Installing Redis..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y redis-server
elif command -v brew &> /dev/null; then
    brew install redis
elif command -v yum &> /dev/null; then
    sudo yum install -y redis
else
    echo "⚠️  Please install Redis manually for your system"
fi

# Start Redis service
echo "🔄 Starting Redis service..."
if command -v systemctl &> /dev/null; then
    sudo systemctl start redis
    sudo systemctl enable redis
elif command -v brew &> /dev/null; then
    brew services start redis
else
    redis-server --daemonize yes
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run Django migrations
echo "🗄️  Running Django migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
echo "👤 Creating Django superuser (optional)..."
echo "You can skip this by pressing Ctrl+C"
python manage.py createsuperuser || echo "Skipped superuser creation"

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the Django WebSocket server:"
echo "   daphne -b 0.0.0.0 -p 8000 baby_monitor.asgi:application"
echo ""
echo "🔗 WebSocket will be available at:"
echo "   ws://localhost:8000/ws/baby-monitor/{baby_id}/"
echo ""
echo "📱 Next.js environment variables needed:"
echo "   NEXT_PUBLIC_WS_URL=ws://localhost:8000"
echo "   NEXT_PUBLIC_DJANGO_API_URL=http://localhost:8000"
