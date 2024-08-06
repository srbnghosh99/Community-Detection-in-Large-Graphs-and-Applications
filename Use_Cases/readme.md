## Install libraries & modules

python --version ---> Python 3.8.16

node --version ---> v18.17.1

## Library installation Run Commands

## How This Works

1. **Install node**
   ```bash
     chmod +x install_node.sh 
     ./install_node.sh 
     ```
2. **Setup and Dependencies**:
   - (optional) set up python virtual environmnet
      - ```python -m venv myenv
        . ./myenv/bin/activate # you will nee to run that script in each termainl session you are using```
        
   - Ensure all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```
4. **Install ngraph modules**
   ```bash
     chmod +x install_node.sh 
     ./install_ngraph.sh
     ```
5. **Go to directory**
   ```bash
     cd <use_case_directory>
     ```
   
## Application execution

Each application folder contains a bash script. Running the script will:

1.  Download the dataset.

2.  The communities file generated from previous steps, is used to analyze the specific tasks of the application.
