#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:58:59 2023

@author: shrabanighosh
"""

import pandas as pd 

df = pd.read_csv("/content/x.edgelist",sep = ' ')
df1 = df['Author1']
df2 = df['Author2']
frames = [df1, df2]
result = pd.concat(frames)
df3 = pd.DataFrame(result)
author_namelist = df3[0].drop_duplicates().tolist()
author_idlist = list(range(1,len(author_namelist)+1)) 
res = {author_namelist[i]: author_idlist[i] for i in range(len(author_namelist))}
df=df.replace({"Author1": res})
df=df.replace({"Author2": res})
df.to_csv("/content/filename.csv")
