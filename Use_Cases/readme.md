## Install libraries & modules

To execute the use cases these modules and libraries required.

python --version ---> Python 3.8.16

node --version ---> v18.17.1

## Library installation Run Commands

## How This Works

This codebase implements the concepts from the paper "A Community-Based Collaborative Filtering Method for Social Recommender Systems" with some modifications to suit our specific use case.

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

Each application folder contains a bash script. Running this script will:

1.  Download the test dataset from the provided link.

2.  Generated from previous step, communities file in JSON/CSV format using different community detection algorithms is then used to analyze the specific tasks of the application.
