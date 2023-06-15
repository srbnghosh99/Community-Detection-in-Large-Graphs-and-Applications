import json
import pandas as pd


with open('author_cluster_memberships.json') as json_file:
    data = json.load(json_file)

authorlis = []
comm_lis = []
for key, value in data.items():
    # print(key, '->', value)
    authorlis.append(key)
    # print(value)
    comm_lis.append(value)
    # break
# len(comm_lis)
# print(len(comm_lis[0]))
df = pd.DataFrame(
    {'author': authorlis,
     'community': comm_lis
    })
df.to_csv("ego_splitting_comm.csv", sep = ' ', index = False)

df = pd.read_csv("ego_splitting_comm.csv", sep = ' ')
df1 = pd.read_csv('/Users/shrabanighosh/Downloads/data/sorted_dblp_author_nameid.csv',sep = ' ')
author_dic = dict(zip(df1.Id, df1.Author))
df['author'] = df['author'].map(author_dic)
df['number_of_communities_author_belong']  = df['community'].apply(lambda x: len(x))
print("The maximum number of communities one author belongs to is ", df.number_of_communities_author_belong.max())
index = df['number_of_communities_author_belong'].idxmax()
print("The author name is",   df['author'][index])

