#!/bin/bash
#
# Launch script for the FastAPI-based dense retrieval server.
#
# Usage:
#     bash launch_server.sh [data_dir] [port] [workers] [reload]
#
# Examples:
#     # Basic usage with defaults
#     bash launch_server.sh
#
#     # Custom data directory and port
#     bash launch_server.sh /path/to/data 8080
#
#     # Production mode with 4 workers
#     bash launch_server.sh /path/to/data 8080 4
#
#     # Development mode with auto-reload
#     bash launch_server.sh /path/to/data 8080 1 reload
#
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH: $PYTHONPATH"
# Parse arguments
DATA_DIR=${1:-"/mnt/public/sunjinfeng/data/search_data/prebuilt_indices"}
PORT=${2:-2727}
WORKERS=${3:-10}


echo "=========================================="
echo "FastAPI Dense Retrieval Server"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"
echo "Workers: $WORKERS"
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

# Build command arguments
CMD_ARGS="--data_dir \"$DATA_DIR\" --port $PORT --host 0.0.0.0 --workers $WORKERS"



# Start server
echo "🚀 Launching FastAPI retrieval server..."
echo "📍 Server will be available at: http://0.0.0.0:$PORT"
echo "📚 API documentation at: http://0.0.0.0:$PORT/docs"
echo ""

eval python retrieval/server.py $CMD_ARGS

echo ""
echo "Server stopped." 
