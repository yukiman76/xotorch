#!/bin/bash
git pull

# Check if node ID was provided as command line argument
if [ $# -eq 0 ]; then
    # If no argument provided, prompt user for node ID
    read -p "Enter the node ID: " node_id
else
    # Use the first command line argument as node ID
    node_id="$1"
fi

# Validate that input is not empty
if [ -z "$node_id" ]; then
    echo "Error: Node ID cannot be empty"
    exit 1
fi

source .venv/bin/activate
# Run the exo command with DEBUG=3 and the provided node ID
DEBUG=3 xot --disable-tui --node-id "$node_id" --default-model "llama-3.2-1b" --inference-engine "torch"