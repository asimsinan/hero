#!/bin/bash
# Script to generate figures from Mermaid diagrams

# Check if mermaid-cli is installed
if ! command -v mmdc &> /dev/null; then
    echo "Mermaid CLI not found. Installing..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Generate System Architecture Diagram
echo "Generating System Architecture Diagram..."
mmdc -i system_architecture.mmd -o system_architecture.pdf -b white -w 1200 -H 800
mmdc -i system_architecture.mmd -o system_architecture.png -b white -w 1200 -H 800

# Generate HNSW Query Process Diagram
echo "Generating HNSW Query Process Diagram..."
mmdc -i hnsw_query_process.mmd -o hnsw_query_process.pdf -b white -w 1400 -H 1000
mmdc -i hnsw_query_process.mmd -o hnsw_query_process.png -b white -w 1400 -H 1000

echo "Done! Generated all figures:"
echo "  - system_architecture.pdf/png"
echo "  - hnsw_query_process.pdf/png"
echo "The PDFs are ready to use in LaTeX."

