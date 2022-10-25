# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NUM_CLASSES = 2
from bs4 import BeautifulSoup
import logging
import json
import os
import random
from transformers import DataProcessor, InputExample
import numpy as np
logger = logging.getLogger(__name__)
#data processor labeled data
class AppReviewProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        #Baby
        examples1 = []
        toy=[]
        word=[]
        file1=open("data/baby_consist.json",'r',encoding='utf-8')
        for line in file1.readlines():
            dic = json.loads(line)
            toy.append(dic)    
        for i in toy:
            word.append(len(i['reviewText']))
        for (i, line) in enumerate(toy):           
            guid = "%s-%s" % ("train", i)
            if line['overall']>=4:
                label='正評'
            elif line['overall']==3:
                label='中立'
            else:
                label='負評'
            examples1.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label=label))
        random.Random(42).shuffle(examples1)        
        #sport
        """See base class."""
        examples2 = []
        toy=[]
        word=[]
        file1=open("data/sport_consist.json",'r',encoding='utf-8')
        for line in file1.readlines():
            dic = json.loads(line)
            toy.append(dic)    
        for i in toy:
            word.append(len(i['reviewText']))
        for (i, line) in enumerate(toy):           
            guid = "%s-%s" % ("train", i)
            if line['overall']>=4:
                label='正評'
            elif line['overall']==3:
                label='中立'
            else:
                label='負評'
            examples2.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label=label))
        random.Random(42).shuffle(examples2)
        #toy
        """See base class."""
        examples3 = []
        toy=[]
        word=[]
        file1=open("data/toy_consist.json",'r',encoding='utf-8')
        for line in file1.readlines():
            dic = json.loads(line)
            toy.append(dic)    
        for i in toy:
            word.append(len(i['reviewText']))
        for (i, line) in enumerate(toy):           
            guid = "%s-%s" % ("train", i)
            if line['overall']>=4:
                label='正評'
            elif line['overall']==3:
                label='中立'
            else:
                label='負評'
            examples3.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label=label))
        random.Random(42).shuffle(examples3)
        #kkbox
        lines = self._read_tsv("data/new_artist.tsv")
        examples4 = []
        a=0
        b=0
        c=0
        random.Random(10).shuffle(lines)
        for (i, line) in enumerate(lines[:792]):
            
            guid = "%s-%s" % ("train", i)
            label, texta,textb= line[2], line[0],line[4]
            texta=texta.replace('<NE>','')
            texta=texta.replace('</NE>','')
            textb=textb.replace('<NE>','')
            textb=textb.replace('</NE>','')
            #if label not in self.get_labels():
             #   label = '0'
            if label!='正評' and label!='中立' and label!='負評':
                continue
            examples4.append(InputExample(guid=guid, text_a=texta,text_b=textb, label=label))
            if label=='正評':
                a+=1
            if label=='中立':
                b+=1
            if label=='負評':
                c+=1

        
        ptt_negative=[]
        import csv
        with open('data/kkbox_negative_review.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                ptt_negative.append(row)
        for index,i in enumerate(ptt_negative[:57]):
            guid = "%s-%s" % ("train", index+792)
            i[0]=i[0].replace('\n','')
            examples4.append(InputExample(guid=guid, text_a=i[0],text_b=i[1], label='負評'))
        random.Random(42).shuffle(examples4)
        examples=examples1+examples2+examples3
        random.Random(42).shuffle(examples)
        #examples1: baby
        #examples2: sport
        #examples3: toy
        #examples4: KKBOX
        return examples4
    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv( "data/new_artist.tsv")
        text=[]
        examples = []
        examples1=[]
        examples2=[]
        a=0
        b=0
        c=0
        random.Random(10).shuffle(lines)
        for (i, line) in enumerate(lines[791:]):
            if i == 0:	#	Skip the header
                continue
            guid = "%s-%s" % ("train", i)
            label, texta,textb= line[2], line[0],line[4]
            texta=texta.replace('<NE>','')
            texta=texta.replace('</NE>','')
            textb=textb.replace('<NE>','')
            textb=textb.replace('</NE>','')
            #if label not in self.get_labels():
             #   label = '0'
            examples.append(InputExample(guid=guid, text_a=texta,text_b=textb, label=label))
            if label=='正評':
                a+=1
            if label=='中立':
                b+=1
            if label=='負評':
                c+=1
        ptt_negative=[]
        import csv
        with open('data/kkbox_negative_review.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                ptt_negative.append(row)
        for index,i in enumerate(ptt_negative[57:]):
            guid = "%s-%s" % ("train", index+198)
            i[0]=i[0].replace('\n','')
            examples.append(InputExample(guid=guid, text_a=i[0],text_b=i[1], label='負評'))
        random.Random(42).shuffle(examples)
        for i in examples:
            i=json.loads(i.to_json_string())
            text.append(i)    

        return examples
        
    def get_labels(self):
        """See base class."""
        return ['正評','中立','負評']

#data processor for ptt unlabeled artist review
class UnsupAppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        lines=self._read_tsv("data/ptt.tsv")
        examples = []
        i = 0 
        for (i,line) in enumerate(lines):
            i += 1
            guid = "%s-%s" % ("unsup", i)
            texta,textb=line[0],line[1]
            examples.append(InputExample(guid=guid, text_a=texta,text_b=textb,label='負評'))
        random.Random(42).shuffle(examples)
        return examples

appreview_processors = {
    "appreview": AppReviewProcessor,
    "unsup":  UnsupAppReviewPretrainProcessor,
}

appreview_output_modes = {
    "appreview": "classification",
}

appreview_tasks_num_labels = {
    "appreview": NUM_CLASSES,
}
