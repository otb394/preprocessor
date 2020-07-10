#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from src import Preprocess
from src.Preprocess import *
import my_tfidf
from my_tfidf import extract
import csv
from collections import Counter

data_path = os.getcwd()

input_file = 'dataset/pitsF.csv'
input_file_path = data_path + '/' + input_file

processed_file_output = 'dataset/proc_pitsF.txt'
proc_file_path = data_path + '/' + processed_file_output

final_output_file = 'dataset/final_pitsF.txt'
final_output_path = data_path + '/' + final_output_file

counter = 0

sev = []

exceptions = []

rowCount = 0
with open(input_file_path, 'rU', encoding='utf-8', errors='ignore') as ifil:
    reader = csv.DictReader(ifil)
    for row in reader:
        #print(row)
        try:
            sev.append(int(row['Severity']))
        except:
            sev.append(0)
            exceptions.append(rowCount)
        rowCount += 1

sevMap = Counter(sev)


#with open(text_info_file_path, 'r') as f:
with open(input_file_path, 'rU', encoding='utf-8', errors='ignore') as f:
    with open(proc_file_path, 'w') as to:
        for line in f:
        #for line in f.readlines():
            if (counter == 0):
                counter += 1
                continue
            x = process(line, string_lower, email_urls, unicode_normalisation, punctuate_preproc,numeric_isolation, stopwords, stemming, word_len_less, str_len_less)
            if len(x) > 1:
                to.write(x)
                to.write('\n')
            counter = counter + 1

print('counter = ' + str(counter))

extract(proc_file_path, final_output_path)

feature_file = 'dataset/feature_pitsF.csv'
feature_file_path = data_path + '/' + feature_file

#existing_data = 'dataset/pitsF.txt'
#existing_data_path = data_path + '/' + existing_data

print('Length of sev = ' + str(len(sev)))
print('No of exceptions = ' + str(len(exceptions)))
print('Exceptions = ' + str(exceptions))

print(sevMap)

#with open(existing_data_path, 'r', encoding='utf-8') as f:
with open(final_output_path, 'r', encoding='utf-8') as f:
    #with open(feature_file_path, 'w') as o:
        corpus = []
        st = set()
        data = []
        for line in f.readlines():
            ws = line.split()
            corpus.append(Counter(ws))
            for w in ws:
                st.add(w)

        lst = list(st)

        rowNum = 0
        for line in corpus:
            row = []
            for w in lst:
                row.append(line[w])
            #print('rowNum = ' + str(rowNum))
            row.append(sev[rowNum])
            data.append(row)
            rowNum += 1

        cols = lst + ['Severity']
        output_df = pd.DataFrame(data, columns=cols)
        output_df.to_csv(feature_file_path, index=False)


