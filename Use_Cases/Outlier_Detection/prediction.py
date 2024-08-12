import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, KFold

def compute_prediction(df,thresholds,weight_values):
    print(thresholds,weight_values)

    T1 = thresholds[0]
    T2 = thresholds[1]
    T3 = thresholds[2]
    T5 = thresholds[3]
    T6 = thresholds[4]
    W1 = weight_values[0]
    W2 = weight_values[1]
    W3 = weight_values[2]
    W5 = weight_values[3]
    W6 = weight_values[4]

    df['I1'] = df['Num_Communities'].apply(lambda x: 1 if x > T1 else 0)
    df['I2'] = df['Community_Ratio'].apply(lambda x: 1 if x > T2 else 0)
    df['I3'] = df['Clustering_Coeff'].apply(lambda x: 1 if x > T3 else 0)
    df['I5'] = df['Clique_Score'].apply(lambda x: 1 if x > T5 else 0)
    df['I6'] = df['Stark_Score'].apply(lambda x: 1 if x > T6 else 0)

    prediction_score = []
    for index, row in df.iterrows():
        score = [W1*row['I1'],W2*row['I2'],W3*row['I3'],W5*row['I5'],W6*row['I6']]
        score = sum(score)
        prediction_score.append(score)

    return prediction_score
    # df['Prediction_score'] = prediction_score


def feature_weight_threshold_estimation(graph_features,labelled_file):

    df = pd.read_csv(graph_features)
    labels = pd.read_csv(labelled_file)
    print(df,labels)
    df = df.merge(labels, how='inner', on='Node')
    df = df.sort_values(by=['Node'])
    # Separate features and target
    X = df[['Num_Communities',
        'Community_Ratio', 'Clustering_Coeff', 'Clique_Score', 'Stark_Score']]
    y = df['Anomaly']
    # df = shuffle(df, random_state=42)
    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=None, random_state=None)

    # Initialize and train XGBoost model

    # clf = lgb.LGBMClassifier()
    model = xgb.XGBClassifier(eval_metric='logloss')

    # Set up K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    print("Average Accuracy across 5 folds:", scores.mean())

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Print the accuracy for each fold
        print("Fold Accuracy:", accuracy_score(y_test, y_pred))
        
        # Get precision, recall, and F1-score for each fold
        report = classification_report(y_test, y_pred, target_names=['Anomaly 0', 'Anomaly 1'])
        print(report)

    # After the cross-validation, if you want to use the entire dataset to train the final model:
    model.fit(X, y)
    # model.fit(X_train, y_train)

    # Make predictions
    # y_pred = model.predict(X_test)
    # print("Accuracy:", accuracy_score(y_test, y_pred))

    # # Get precision, recall, and F1-score
    # report = classification_report(y_test, y_pred, target_names=['Anomaly 0', 'Anomaly 1'])
    # print(report)



    # Extract decision thresholds (splits) from the model
    thresholds = {feature: [] for feature in X.columns}
    booster = model.get_booster()

    for tree in booster.get_dump(with_stats=True):
        for line in tree.splitlines():
            if "<" in line:  # Check if the line contains a threshold split
                parts = line.split("<")
                # feature_index = str(parts[0].split("[")[1].split("]")[0])  # Extract the feature index
                feature_name = str(parts[0].split("[")[1].split("]")[0])
                # print(feature_index)
                threshold_value = float(parts[1].split("]")[0])  # Extract the threshold value
                # feature_name = X.columns[feature_index]  # Map to your feature name
                thresholds[feature_name].append(threshold_value)

    # Calculate the average threshold for each feature
    avg_thresholds = {feature: sum(values)/len(values) if values else None for feature, values in thresholds.items()}
    print("Average Thresholds:", avg_thresholds)


    # Print thresholds
    thresholds = []
    for feature, threshold in avg_thresholds.items():
        if threshold is not None:
            thresholds.append(threshold)
            print(f"Estimated Threshold for {feature}: {threshold:.2f}")
        else:
            thresholds.append(0.0)
            # avg_thresholds[feature] = 0.0
            print(f"No splits found for feature {feature}")


    # Get feature importances
    importances = model.feature_importances_
    weight_values = []
    # Print feature importances
    for feature, importance in zip(X.columns, importances):
        print(f"Feature: {feature}, Importance: {importance}")
        weight_values.append(importance)


    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from XGBoost Model')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.show()

    # Threshold determination (for binary classification)
    # For binary classification, the default threshold is 0.5, but you can tune it based on your specific requirements.
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, thresholds, color='skyblue')
    plt.xlabel('Feature Threshold')
    plt.title('Feature Threshold from XGBoost Model')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.show()



    df['Prediction_score'] = compute_prediction(df,thresholds,weight_values)
    df.to_csv('Prediction_score2.csv')



def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")
    
    parser.add_argument("--graph_features",type = str)
    parser.add_argument("--labelled_file",type = str)
    # parser.add_argument("--outputfile",type = str)
    return parser.parse_args()

def main():
    inputs=parse_args()
    feature_weight_threshold_estimation(inputs.graph_features,inputs.labelled_file)
  
if __name__ == '__main__':
    main()
