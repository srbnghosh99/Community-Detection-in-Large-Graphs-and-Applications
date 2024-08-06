# Community Detection Models and Downstream Tasks

## Overview

This repository provides a comprehensive framework for community detection and downstream task analysis using various models. The workflow involves using community detection models from the `models` directory and applying them to different downstream tasks located in the `Use_Cases` directory. Each application folder contains specific instructions to guide you through the process.

## Steps to Use

1. **Choose a Community Detection Model**
   - Navigate to the `models` directory. You will find subdirectories for different community detection algortihms. Choose which one you want to use. Navigate to that directory and follow the steps listed in the read me file there.

2. **Run the Bash Script**
   - The script performs the following actions:
     1. **Download the Dataset**: Retrieves the dataset from a provided link.
     2. **Generate Communities File**: Creates a communities file in JSON/CSV format using the selected community detection model.

2. **Navigate to the Application Folder**
   - Each application folder contains a bash script that facilitates the entire process. 

4. **Analyze with Downstream Tasks**
   - Utilize the generated communities file to perform analysis as per the specific tasks described in the `Use_Cases` directory.

## Detailed Instructions

- **Models Directory**:
  - This directory contains different community detection algorithms. Select the model that best fits your analysis needs.
  
- **Use Cases Directory**:
  - Each folder in this directory corresponds to a different downstream task. Follow the specific instructions provided in each folder to carry out the analysis.

## Example Workflow

1. **Choose a Community Detection Model**:
   ```bash
   cd models
   # Select the model and follow any setup instructions if required

2. ***Compare communities***
   ```bash
   python3 comm_quality.py --graphfile graph.csv --directory_path path-to-communities-folder

3. ***Plot comparison of communities***
   ```bash
   python3 plot_community_metrices.py --directory_path path-to-communities_metrics-folder
   
3. **Choose downstream task**:
   ```bash
   cd Use_Cases
   # Select the use case and follow setup instructions if required 
