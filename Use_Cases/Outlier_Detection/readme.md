# Anomaly Detection

## Overview

This repository implements a anomaly detection based on the principles outlined in the paper "AD-C: a new node anomaly detection based on community detection in social networks". At a high level, our system uses community detection to enhance the prediction of anomalies.

## How It Works

1. **Community Detection**: 
   - Communities are detected using social graph/financial transaction graph. Different community detections models can be used for this step. 
2. **Identifying Auxiliary Communities**:
   - Overlapping communities are identified using border nodes of non-overlapping communities.
3. **Anomaly Detection**:
   - Anomaly detection is performed by applying XGBoost classification model 
### Evaluation Process

1. **Performance Analysis**: Analyze the results to determine if our system provides significant improvements in prediction based on ground truth anomalies.


## How This Code Works

This codebase implements the concepts from the paper "AD-C: a new node anomaly detection based on community detection in social networks" with some modifications to suit our specific use case.


## Reference

This code is based on the methodologies and concepts presented in the paper:
- "AD-C: a new node anomaly detection based on community detection in social networks"

By following these steps and utilizing the provided scripts, you can implement any trust prediction tailored to your specific needs. Feel free to modify the code to better suit your application and improve the prediction quality.
