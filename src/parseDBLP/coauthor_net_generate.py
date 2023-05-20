#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:58:59 2023

@author: shrabanighosh
"""

import argparse
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from itertools import permutations,combinations


def read_file(filename):
  df = pd.read_csv(filename,sep = ',')
  new_file_name = filename.replace('.csv', '.edgelist') 
  # new_file_name = "data/coauthor_net_" + new_file_name
  # new_file_name_freq = "data/coauthor_net_freq_" + new_file_name

  with open(new_file_name, 'w') as f:
    f.write(f"Author1 Author2\n")
    for i, row in df.iterrows():
      # row['authors_name']
      lis = row['authors_name'].split(',')
      lis = [x.strip(' ') for x in lis]
      # print(lis)
      strippedText = str(lis).replace('[','').replace(']','').replace('\'','').replace('\"','')
      # print(strippedText)
      my_list = strippedText.split(",")
      my_list = [x.strip(' ') for x in my_list]
      my_list = [x.replace(' ', '_') for x in my_list] 
      my_list = [i for i in my_list if i]
      # print(my_list)
      comb = combinations(my_list, 2)
      for i in list(comb):
        f.write(f"{i[0].strip()} {i[1].strip()}\n")
  print("File Generated Co-author Network")
  df1 = pd.read_csv(new_file_name,sep = " ")
  freq = df1.groupby(["Author1", "Author2"]).size().reset_index(name="Weight")
  freq = freq.sort_values(by=['Weight'],ascending=False).reset_index()
  freq['Weight'] = freq[['Weight']].astype(int)
  freq= freq[["Author1","Author2","Weight"]]
  new_file_name = filename.replace('.csv', '_freq.edgelist') 
  freq.to_csv(new_file_name, sep = " ",header = False, index = False)
  print("File Generated Co-author Frequncy Network")

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
