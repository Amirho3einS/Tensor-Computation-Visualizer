#!/bin/bash

# Start the Vite development server with auto-reload (HMR)
# This script runs the vector-mac-visualizer dev server

# Navigate to the directory containing this script
cd "$(dirname "$0")"

# Start the development server
echo "Starting Vector MAC Visualizer development server..."
echo "Auto-reload (HMR) is enabled by default with Vite."
echo "Press Ctrl+C to stop the server."
echo ""

npm run dev
