#!/bin/bash
#
# Launch script for the FastAPI-based dense retrieval server.
#
# Usage:
#     bash launch_server.sh [data_dir] [port] [workers] [mode]
#
# Examples:
#     # Basic usage with defaults
#     bash launch_server.sh
#
#     # Custom data directory and port
#     bash launch_server.sh /path/to/data 8080
#
#     # Production mode with multiple workers (Note: each worker loads model)
#     bash launch_server.sh /path/to/data 8080 2
#
#     # Development mode with auto-reload
#     bash launch_server.sh /path/to/data 8080 1 reload
#
#     # Specify log level
#     LOG_LEVEL=debug bash launch_server.sh
#

# Set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
DATA_DIR=${1:-"/mnt/public/sunjinfeng/data/search_data/prebuilt_indices"}
PORT=${2:-2727}
WORKERS=${3:-10}
MODE=${4:-""}
LOG_LEVEL=${LOG_LEVEL:-"info"}


echo "=========================================="
echo "FastAPI Dense Retrieval Server Launcher"
echo "=========================================="
echo "PYTHONPATH: $PYTHONPATH"
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Mode: ${MODE:-production}"
echo "Log level: $LOG_LEVEL"
echo "=========================================="

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found!"
    echo ""
    echo "Please run download_search_data.py first:"
    echo "  python examples/search/download_search_data.py"
    exit 1
fi

# Check for required files
echo "Checking required files..."
required_files=("../wikipedia/wiki-18.jsonl" "e5_Flat.index")
for file in "${required_files[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "❌ Error: $file not found in $DATA_DIR"
        echo ""
        echo "Please run download_search_data.py to setup data"
        exit 1
    fi
done
echo "✓ All required files present"

# Check if FastAPI dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi, uvicorn, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required dependencies not found!"
    echo ""
    echo "Please install FastAPI dependencies:"
    echo "  pip install fastapi uvicorn pydantic"
    exit 1
fi
echo "✓ All dependencies installed"
echo ""

# Warning about workers
if [ "$WORKERS" -gt 1 ]; then
    echo "⚠️  WARNING: Running with $WORKERS workers"
    echo "⚠️  Each worker will load its own copy of the model and index!"
    echo "⚠️  This will multiply memory usage by $WORKERS"
    echo "⚠️  Consider using 1 worker unless you need high concurrency"
    echo ""
    read -p "Continue with $WORKERS workers? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Build command arguments
CMD_ARGS="--data_dir \"$DATA_DIR\" --port $PORT --host 0.0.0.0 --workers $WORKERS --log-level $LOG_LEVEL"

# Add reload flag if in development mode
if [ "$MODE" = "reload" ] || [ "$MODE" = "dev" ]; then
    CMD_ARGS="$CMD_ARGS --reload"
    if [ "$WORKERS" -gt 1 ]; then
        echo "⚠️  Auto-reload mode requires single worker. Forcing workers=1"
        CMD_ARGS="--data_dir \"$DATA_DIR\" --port $PORT --host 0.0.0.0 --workers 1 --reload --log-level $LOG_LEVEL"
    fi
fi

# Start server
echo ""
echo "🚀 Launching FastAPI retrieval server..."
echo "📍 Server will be available at: http://0.0.0.0:$PORT"
echo "📚 API documentation at: http://0.0.0.0:$PORT/docs"
echo "🏥 Health check at: http://0.0.0.0:$PORT/health"
echo ""
echo "Starting with command:"
echo "python retrieval/server.py $CMD_ARGS"
echo ""

# Navigate to project root and start server
cd "$PROJECT_ROOT" || exit 1
eval python retrieval/server.py $CMD_ARGS

echo ""
echo "Server stopped." 
