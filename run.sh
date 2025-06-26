#!/bin/bash
git pull
# Prompt user for node ID
read -p "Enter the node ID: " node_id

# Validate that input is not empty
if [ -z "$node_id" ]; then
    echo "Error: Node ID cannot be empty"
    exit 1
fi

source .venv/bin/activate
# Run the exo command with DEBUG=9 and the provided node ID
# DEBUG=2 
DEBUG=3 xot --disable-tui --node-id "$node_id"  --default-model "llama-3.2-1b" --inference-engine "torch"