# Trust Prediction

## Overview

This repository implements a recommendation system based on the principles outlined in the paper "Leveraging Community Detection for Accurate Trust Prediction". At a high level, our system uses community detection to enhance the prediction of trust relationships, which in turn improves the accuracy of recommendations by leveraging trust among users.

## How It Works

## How It Works

1. **Community Detection**: 
   - Communities are detected using trust networks. Different community detections models can be used for this step. 
   
2. **Identifying Community Centers**:
   - Community centers are identified using centrality measures to find representative users within each community.
   
3. **Trust Prediction**:
   - Trust prediction is performed by selecting corresponding communities from the users’ membership vectors that 1) are similar to each other and 2) match the users well, as measured by similarity between the users and the community centers.

### Evaluation Process

1. **Performance Analysis**: Analyze the results to determine if our system provides significant improvements in prediction based on ground truth trust relations.


## How This Code Works

This codebase implements the concepts from the paper "A Community-Based Collaborative Filtering Method for Social Recommender Systems" with some modifications to suit our specific use case.

1. **Overlapping Communitiess**:
   - Ensure the output file of communities in json format:
     ```bash
     ./script.sh
     ```
     
2. **Non-Overlapping Communities**:
   - Ensure the output file of communities in csv format:
     ```bash
     ./script.sh

## Reference

This code is based on the methodologies and concepts presented in the paper:
- "Leveraging Community Detection for Accurate Trust Prediction"

By following these steps and utilizing the provided scripts, you can implement any trust prediction tailored to your specific needs. Feel free to modify the code to better suit your application and improve the prediction quality.

