#!/bin/bash
  
# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Verify Node.js installation
echo "Node.js version:"
node --version
echo "npm version:"
npm --version

# Install ngraph.fromjson package
echo "Installing ngraph.fromjson..."
npm install ngraph.fromjson
npm install ngraph.centrality
echo "Installation complete"

~                                                                                                                                                               
~      
