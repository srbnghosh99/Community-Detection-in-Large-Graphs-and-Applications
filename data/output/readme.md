## Output 

# Non-overlapping
The format of the output csv file is a standard space separated node vs community:

```
node community
```

where node holds unique authors_id and community column shows the community_number of the author node belongs to.

The format of the output csv file to show the number of nodes for each community:

```
community number_of_nodes
```

# Overlapping
The format of the output csv file is a standard space separated node vs communities:

```
node communities
```

where node holds unique authors_id and community column shows the list of communities the author node belongs to.

## Louvain 

The output is non-overlapping community information

output format --> node community

There are total 89388 number of communities generated for graph of collaboration freq greater than 1

Total number of communites 58241

## Ego Splitting

The output is overlapping community information

output format --> node communities

Number of nodes 1216501

Total number of communites 1353713

