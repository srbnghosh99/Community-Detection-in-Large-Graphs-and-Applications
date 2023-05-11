#!/usr/bin/env python
import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from itertools import permutations,combinations


def read_file(filename):
  print("main",filename)
  df = pd.read_csv(filename,sep = ',')
  pid_col = df[['info/authors/author/0/_pid','info/authors/author/1/_pid','info/authors/author/2/_pid','info/authors/author/3/_pid','info/authors/author/4/_pid','info/authors/author/5/_pid','info/authors/author/6/_pid','info/authors/author/7/_pid','info/authors/author/8/_pid','info/authors/author/9/_pid','info/authors/author/10/_pid','info/authors/author/11/_pid','info/authors/author/12/_pid','info/authors/author/13/_pid','info/authors/author/14/_pid','info/authors/author/15/_pid','info/authors/author/16/_pid','info/authors/author/17/_pid','info/authors/author/18/_pid','info/authors/author/19/_pid']]
  pid_col.dropna(axis=1,how='all')
  with open('coauthor_net.txt', 'w') as f:
    f.write(f"A1 A2\n")
    for i, row in pid_col.iterrows():
        lis = []
        for j, column in row.iteritems():
          if(column is not np.nan):
              lis.append(column)
        # print((lis))
        comb = combinations(lis, 2)
        for i in list(comb):
          # text = (i)
          f.write(f"{i[0]} {i[1]}\n")
  df1 = pd.read_csv("coauthor_net.txt",sep = " ")
  df2 = df1['A1']
  df3 = df1['A2']
  frames = [df2, df3]
  result = pd.concat(frames)
  df4 = pd.DataFrame(result)
  author_id = df4[0].unique().tolist()
  lis = list(range(1,len(author_id)))
  len(author_id),len(lis)
  res = res = dict(zip(author_id, lis))
  df1['A3'] = df1['A1'].map(res)
  df1['A4'] = df1['A2'].map(res)
  freq = df1.groupby(["A3", "A4"]).size().reset_index(name="Weight")
  freq = freq.sort_values(by=['Weight'],ascending=False).reset_index()
  freq= freq[["A3","A4","Weight"]]
  freq['A4'] = freq[['A4']].astype(int)
  # print(freq)
  new_file_name = filename.replace('.csv', '.edgelist') 
  new_file_name = "coauthor_net_" + new_file_name
  new_file_name = "data/" + new_file_name
  print(new_file_name)
  freq.to_csv(new_file_name, sep = " ",header = False, index = False)



def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--filename",type = str)
    
    return parser.parse_args()

def main():

    
    inputs=parse_args()
    print(inputs.filename)
    read_file(inputs.filename)
  

if __name__ == '__main__':
    main()
