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
   - (optional) set up python virtual environmnet
      - ```
        /path/to/the/right/version/of/python -m venv myenv
        . ./myenv/bin/activate # you will nee to run that script in each termainl session you are using```
        
   - Ensure all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```
4. **Install ngraph modules**
   ```bash
     ./install_ngraph.sh
     ```
5. **Go to directory**
   ```bash
     cd <use_case_directory>
     ```
   
## Application execution

Each application folder contains a bash script. (see detail in the applicatino readme file.) Running the script will:

1.  Download the dataset.

2.  The communities file generated from previous steps, is used to analyze the specific tasks of the application.
