# Community Detection Models and Downstream Tasks

## Overview

This repository provides a comprehensive framework for community detection and analysis using various models. The workflow involves using community detection models from the `models` directory and applying them to different downstream tasks located in the `Use_Cases` directory. Each application folder contains specific instructions to guide you through the process.

## Steps to Use

1. **Choose a Community Detection Model**
   - Navigate to the `models` directory and select the appropriate community detection algorithm based on your requirements.

2. **Run the Bash Script**
   - The script performs the following actions:
     1. **Download the Test Dataset**: Retrieves the dataset from a provided link.
     2. **Generate Communities File**: Creates a communities file in JSON/CSV format using the selected community detection model.

2. **Navigate to the Application Folder**
   - Each application folder contains a bash script that facilitates the entire process. 

4. **Analyze with Downstream Tasks**
   - Utilize the generated communities file to perform analysis as per the specific tasks described in the `Use_Cases` directory.

## Detailed Instructions

- **Models Directory**:
  - This directory contains different community detection algorithms. Select the model that best fits your analysis needs.
  
- **USE Cases Directory**:
  - Each folder in this directory corresponds to a different downstream task. Follow the specific instructions provided in each folder to carry out the analysis.

## Example Workflow

1. **Choose a Community Detection Model**:
   ```bash
   cd models
   # Select the model and follow any setup instructions if required

2. cd Use_Cases
   ```bash
   cd Use_Cases
   # Select the use case and follow setup instructions if required 
