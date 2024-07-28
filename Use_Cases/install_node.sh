# Step 1: Install nvm (if not already installed)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash

# Step 2: Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
source ~/.bashrc  # or source ~/.zshrc

# Step 3: Verify nvm installation
nvm --version

# Step 4: Install Node.js version 18.17.1
nvm install 18.17.1

# Step 5: Use the installed Node.js version
nvm use 18.17.1
nvm alias default 18.17.1

# Step 6: Verify Node.js installation
node --version
