## Install libraries & modules

python --version ---> Python 3.8.16

node --version ---> v18.17.1

## Library installation Run Commands

## How This Works

1. **Install node**
   ```bash
     ./install_node.sh 
     ```
2. **Setup and Dependencies**:
   - Ensure all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```
3. **Install ngraph modules**
   ```bash
     ./install_ngraph.sh
     ```
   
## Application execution

Each application folder contains a bash script. Running the script will:

1.  Download the dataset from the provided link.

2.  The communities file generated from previous steps, is used to analyze the specific tasks of the application.
