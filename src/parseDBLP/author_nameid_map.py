#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:58:59 2023

@author: shrabanighosh
"""

import pandas as pd 
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def author_nameid_map(inputfile,outputfile):
    df = pd.read_csv(inputfile,sep = ' ')
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
    # df.to_csv("/content/filename.csv")
    df.to_csv(outputfile,index = False,sep = ' ', header= False)

def parse_args():
   
    parser = argparse.ArgumentParser(description="Read File")

    parser.add_argument("--inputfile",type = str)
    parser.add_argument("--outputfile",type = str)
    
    return parser.parse_args()

def main():
    inputs=parse_args()
    print(inputs.inputfile)
    print(inputs.outputfile)
    author_nameid_map(inputs.inputfile,inputs.outputfile)
  

if __name__ == '__main__':
    main()
