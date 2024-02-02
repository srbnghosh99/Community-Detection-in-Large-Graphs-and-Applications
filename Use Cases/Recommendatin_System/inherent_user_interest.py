#  <#Title#>

import json
import pandas as pd
import os
import argparse
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np
import ast
import itertools
from surprise.model_selection import cross_validate, KFold
from sklearn.utils import shuffle

rating = pd.read_csv("ciao_rating.csv")
detected_community_df = pd.read_csv("ciao_trustnet_ego_splitting_membership.csv")
#print(rating)
#print(rating["productid"].nunique())
#print(rating["categoryid"].nunique())
#print(rating["rating"].unique())




detected_community_df['Community'] = detected_community_df['Community'].apply(ast.literal_eval)
#print("detected_community_df")
#print(detected_community_df)
user_communities = detected_community_df.to_dict()

user_communities = detected_community_df.set_index('Node')['Community'].to_dict()

community_mapping = {}
for index, row in detected_community_df.iterrows():
    nodes = row['Node']
    community = row['Community']
    # print(community)
    for c in community:
        # print(c)
        if c in community_mapping:
            community_mapping[c].append(nodes)
        else:
            community_mapping[c] = [nodes]

#print(community_mapping[0])
#community_mapping_df = pd.DataFrame.from_dict(community_mapping, orient='index')
community_mapping_df = pd.DataFrame(list(community_mapping.items()), columns=['Community', 'Nodes'])
community_mapping_df = community_mapping_df.sort_values(by='Community', ascending=True)
#print(community_mapping_df)


community_preference_vector = {}
for index, row in community_mapping_df.iterrows():
    comm = row['Community']
    node = row['Nodes']
    trunc_rating = rating[rating['userid'].isin(node)]
#    print(trunc_rating['rating'].mean())
    community_preference_vector[comm]=trunc_rating['rating'].mean()
#    break


community_preference_vector_df = pd.DataFrame(list(community_preference_vector.items()), columns=['Community', 'Rating_mean'])

print(community_preference_vector_df)

community_preference_vector_df.to_csv("community_preference_vector_df.csv",index = False)




# Assuming 'ratings_df' is your ratings DataFrame
# Replace 'userid' and 'rating' with your actual column names
user_interests = rating.groupby('userid')['rating'].mean()

# If you want to reset the index and obtain a DataFrame
user_interests_df = user_interests.reset_index()

# Rename the columns for clarity (optional)
user_interests_df.columns = ['userid', 'inherent_interest']

# Display the User Interest Vector
print(user_interests_df)

user_interests_df.to_csv("user_interests_df.csv",index = False)

#print(user_communities)
# Convert DataFrame to dictionary
#user_communities = detected_community_df.to_dict()
#user_communities = {
#    'user1': ['communityA', 'communityB'],
#    'user2': ['communityB', 'communityC'],
#    # ...
#}
#print(user_communities[2])

user_interests_df = pd.read_csv("user_interests_df.csv")
community_preference_vector_df = pd.read_csv("community_preference_vector_df.csv")
#column_names = ['userid', 'Closeness', 'SameAsDegreeCentrality','Betweenness']
node_propensity_df = pd.read_csv("node_propensity_dataframe.csv")
#print(user_interests_df)
#print(community_preference_vector_df)
#print(node_propensity_df)





#print(filtered_testset)
'''

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userid', 'itemid', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define the SVD model
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model (optional)
accuracy.rmse(predictions)

# Example: Get predictions for a specific user and item using the equation
user_id = 1
item_id = 100

# Get user bias (Bu), item bias (Bi), and latent vectors (Qi, Pu)
bu = model.bu[user_id]
bi = model.bi[item_id]
qi = model.qi[item_id]
pu = model.pu[user_id]


'''

node_propensity_df['Closeness'] = node_propensity_df['Closeness'].apply(ast.literal_eval)
node_propensity_df['Community'] = node_propensity_df['Community'].apply(ast.literal_eval)


        


# Calculate the predicted rating using the equation
#predicted_rating = model.trainset.global_mean + bu + bi + np.dot(qi, pu)
#print(f'Predicted rating for user {user_id} on item {item_id}: {predicted_rating}')


# Load your dataset
#reader = Reader(line_format='user item rating community', sep=' ')
#data = Dataset.load_from_file('your_dataset.csv', reader=reader)
#data = rating
# Split the data into training and testing sets

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(rating[['userid', 'productid', 'rating']], reader)
trainset = data.build_full_trainset()
num_users = trainset.n_users
num_items = trainset.n_items

print(f"Number of users in the trainset: {num_users}")
print(f"Number of items in the trainset: {num_items}")

## Use SVD model (you can choose a different model)
model = SVD()
model.fit(trainset)



## the nodes in the rating dataset, all are not included in the social graph. that is why we excluded the nodes are not in the social graph
# from test dataset because we do not have propensity values of the nodes not in social graph.
ciao = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_trustnet.csv",sep = ' ')

ciao.rename(columns={ciao.columns[0]:'Node1',ciao.columns[1]:'Node2'}, inplace=True)
col1 = ciao['Node1'].tolist()
col2 = ciao['Node2'].tolist()
nodelis = col1 + col2

rating = pd.read_csv("/Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_rating.csv",sep = ',')
list2 = list(rating['userid'].unique())
list1 = list(rating['productid'].unique())
values_to_exclude = list(set(list2) - set(nodelis))
#print(len(values_to_exclude))


# Exclude sets with matching first values
_, testset = train_test_split(data, test_size=0.2, random_state=42)
filtered_testset = {item for item in testset if item[0] not in values_to_exclude}
#print(filtered_testset)

#for userid in list2:
#    item_bias = model.bi[productid]
#    print(userid, user_bias)
#for productid in list1:
#    item_bias = model.bi[productid]
#    print(productid, item_bias)
#
#bu_last_user = model.bu[trainset.to_inner_uid(7374)]
#bi_last_item = model.bi[trainset.to_inner_iid(105113)]
#item_bias = model.bi[105113]
#user_bias = model.bu[7374]
#print(bu_last_user, bi_last_item,item_bias,user_bias  )

# data = Dataset.load_from_df(filtered_testset[['userid', 'productid', 'rating']], reader)
# kf = KFold(n_splits=5, random_state=42, shuffle=True)
actual_ratings = []
predicted_ratings = []
kf = KFold(n_splits=5, random_state=42, shuffle=True)
predicted_ratings_set = set()
    
for trainset2, testset2 in kf.split(data):
    # Shuffle the test set for each iteration
    shuffled_testset = testset2

    for userid, productid in set((user, item) for user, item, rating in shuffled_testset):
        actual_ratings.append(rating)
        is_userid_in_dataframe = userid in node_propensity_df['userid'].values
        if is_userid_in_dataframe is False:
            print(userid)
            continue
        communities = node_propensity_df[node_propensity_df['userid'] == userid]['Community'].iloc[0]
        # print(communities)
        propensity = node_propensity_df[node_propensity_df['userid'] == userid]['Closeness'].iloc[0]
        result = 0
        for index, prop in enumerate(propensity):
            comm = communities[index]
            result += prop * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
        inner_product_id = trainset.to_inner_iid(productid)
        inner_user_id = trainset.to_inner_uid(userid)
        bu = model.bu[inner_user_id]
        bi = model.bi[inner_product_id]
        qi = model.qi[inner_product_id]
        pu = model.pu[inner_user_id]
        result = pu + result
        qi_transposed = qi.T
        predicted_rating = model.trainset.global_mean + bu + bi + np.dot(qi_transposed, result)
        predicted_ratings.append(predicted_rating)
        # print(predicted_ratings)
        predicted_ratings_set.add((userid, productid, predicted_rating))



# Now compare predicted ratings with actual ratings in shuffled_testset
# Calculate RMSE
    # rmse = accuracy.rmse(predicted_ratings_set, verbose=False)
    # print(f"RMSE for this fold: {rmse}")
print("Prediction completed")
# Calculate overall RMSE
overall_rmse = accuracy.rmse(predicted_ratings, actual_ratings, verbose=False)
print(f"Overall RMSE across all folds: {overall_rmse}")

# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(filtered_testset[['userid', 'productid', 'rating']], reader)

# kf = KFold(n_splits=5, random_state=42, shuffle=True)
# cross_val_results = cross_validate(model, data, measures=['RMSE'], cv=kf, verbose=True)

# # Print the cross-validation results
# for key, value in cross_val_results.items():
#     print(f"{key}: {sum(value) / len(value)}")


# # Perform 5-fold cross-validation with your custom test function
# kf = KFold(n_splits=5, random_state=42, shuffle=True)
# cross_val_results = cross_validate(model, data, measures=['RMSE'], cv=kf, verbose=True, test_options={'test': custom_test})

# # Print the cross-validation results
# for key, value in cross_val_results.items():
#     print(f"{key}: {sum(value) / len(value)}")

# Print the set
# print(predicted_ratings_set)

# productid = 106645
# userid = 7319
# predicted_rating = model.predict(userid, productid).est
# print(predicted_rating)

# print(predicted_rating)

# raw_ratings = data.raw_ratings
# threshold = int(0.9 * len(raw_ratings))  # 90% of the data for training
# train_raw_ratings = raw_ratings[:threshold]
# test_raw_ratings = raw_ratings[threshold:]
# testset = test_data.construct_testset(test_raw_ratings)  # Convert the raw test ratings to a testset
# predictions = model.test(testset)
# accuracy.rmse(predictions)
        # Make prediction for the current user-item pair
#        predicted_rating = model.predict(user_id, item_id).est
#        print(f'Predicted rating for user {user_id} on item {item_id}: {predicted_rating}')

# Perform 5-fold cross-validation
# cross_val_results = cross_validate(model, testset, measures=['RMSE'], cv=5, verbose=True)

# # Print the cross-validation results
# for key, value in cross_val_results.items():
#     print(f"{key}: {sum(value) / len(value)}")




#for userid, productid in unique_user_item_pairs:



#for index, row in node_propensity_df.iterrows():
#    user_id = row['userid']
#    communities = row['Community']
#    propensity = row['Closeness']
#    print(propensity)
#    comm = communities[index]
#    print(comm)
#    for index, prop in enumerate(propensity):
#        result = prop * community_preference_vector_df[community_preference_vector_df['Community'] == comm]['Rating_mean'].iloc[0]
#        print(result)


# Get latent vectors
 
# Assuming 'model' is your trained SVD model
#μ is the global mean of all ratings
#user_bias_vectors = model.bu
#item_bias_vectors = model.bi
#item_latent_vectors = model.qi

#for user_id, item_id in testset.unique_pairs():
#    # Make prediction for the current user-item pair
#    predicted_rating = model.predict(user_id, item_id).est
#    print(f'Predicted rating for user {user_id} on item {item
#
#predicted_rating = model.trainset.global_mean + bu + bi + np.dot(qi, pu)
#
## Example: Predict the rating for user 'user_id' on item 'item_id'
#user_id = 'user_id'
#item_id = 'item_id'
#predicted_rating = model.predict(user_id, item_id).est
#
#
## You may need to incorporate community information in a different way
## For example, by creating a community latent vector for each community
## based on the average of item vectors associated with that community.
#community_latent_vectors = item_latent_vectors.groupby(rating['community']).mean().values

# Now, user_latent_vectors, item_latent_vectors, and community_latent_vectors
# contain the latent vectors for users, items, and communities, respectively





#community_latent_vectors = {}
#for user_id, communities in user_communities.items():
#    user_vector = model.pu[model.trainset.to_inner_uid(user_id)]
#    for community in communities:
#        if community not in community_latent_vectors:
#            community_latent_vectors[community] = []
#        community_latent_vectors[community].append(user_vector)
#
## Calculate mean latent vector for each community
#for community, vectors in community_latent_vectors.items():
#    community_latent_vectors[community] = np.mean(vectors, axis=0)





#
## Train the model
#model.fit(trainset)
#
## Make predictions on the test set
#predictions = model.test(testset)
#
## Evaluate the model (optional)
#accuracy.rmse(predictions)



#user_id = 1



#unique_user_item_pairs = set(itertools.product(rating['userid'].unique(), rating['productid'].unique()))
#print(unique_user_item_pairs)
#unique_data = Dataset.load_from_df(pd.DataFrame(list(unique_user_item_pairs), columns=['userid', 'itemid', 'rating']), reader)

#model = SVD()
#trainset = unique_data.build_full_trainset()
#model.fit(trainset)
#print(testset)
 
