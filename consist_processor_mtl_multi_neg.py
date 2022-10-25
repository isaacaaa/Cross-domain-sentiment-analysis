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
import csv
import sys
from transformers import DataProcessor, InputExample
logger = logging.getLogger(__name__)

# data processor for source domain and target domain labeled data
# source domain: Baby, Sport, Toy; Target domain: KKBOX
class AppReviewProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        # baby domain
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

        # sport domain
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
        
        # toy domain
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
        
        # KKBOX domain
        lines = self._read_tsv("data/new_artist.tsv")
        examples4 = []
        a=0
        b=0
        c=0
        random.Random(10).shuffle(lines)
        for (i, line) in enumerate(lines[:792]):
            guid = "%s-%s" % ("kkbox", i)
            label, texta,textb= line[2], line[0],line[4]
            texta=texta.replace('<NE>','')
            texta=texta.replace('</NE>','')
            textb=textb.replace('<NE>','')
            textb=textb.replace('</NE>','')
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
        with open('data/kkbox_negative_review.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                ptt_negative.append(row)
        for index,i in enumerate(ptt_negative[:57]):
            guid = "%s-%s" % ("kkbox", index+792)
            i[0]=i[0].replace('\n','')
            examples4.append(InputExample(guid=guid, text_a=i[0],text_b=i[1], label='負評'))
        random.Random(42).shuffle(examples4)
        # baby
        # return examples[:4196]
        # sport
        # return examples[:1879]
        # toy
        # return examples[:4236]
        return examples1[:4196],examples2[:1879],examples3[:4236],examples4

    # kkbox testing data    
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
        return examples,text

    def get_labels(self):
        """See base class."""
        return ["正評","中立","負評"]

# data processor for target domain unlabeled data (PTT unlabeled data)
class UnsupAppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        lines=self._read_tsv('data/ptt.tsv')      
        examples = []
        i = 0 
        for (i,line) in enumerate(lines):
            i += 1
            guid = "%s-%s" % ("unsup", i)
            texta,textb=line[0],line[1]
            examples.append(InputExample(guid=guid, text_a=texta,text_b=textb,label="中立"))
        random.Random(42).shuffle(examples)
        return examples

# data processor for relevant detection task         
class SimilarProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        import json
        def rotate(l,n):
            return l[n:]+l[:n]
        baby_name = []
        sport_name = []
        toy_name = []
        file1 = open("data/baby_consist.json", 'r', encoding = 'utf-8')
        for line in file1.readlines():
            dic = json.loads(line)
            baby_name.append(dic['Title'])
        file2 = open("data/sport_consist.json", 'r', encoding= 'utf-8')
        for line in file2.readlines():
            dic = json.loads(line)
            sport_name.append(dic['Title'])
        file3 = open("data/toy_consist.json", "r", encoding = 'utf-8')
        for line in file3.readlines():
            dic = json.loads(line)
            toy_name.append(dic['Title'])
        all_target = baby_name+sport_name+toy_name
        # positive sample
        source_positive_samples = []
        # negative sample
        source_negative_samples = []
        final_source_positive_samples = []
        final_source_negative_samples = []
        # out of domain negative samples (Amazon) & positive samples (Amazon)
        file1=open("data/baby_consist.json",'r',encoding='utf-8')
        baby = []
        for line in file1.readlines():
            dic = json.loads(line)
            baby.append(dic)    
        for (i, line) in enumerate(baby):           
            guid = "%s-%s" % ("train", i)
            source_positive_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label = '有關'))
        file1=open("data/sport_consist.json",'r',encoding='utf-8')
        sport = []
        for line in file1.readlines():
            dic = json.loads(line)
            sport.append(dic)    
        for (i, line) in enumerate(sport):           
            guid = "%s-%s" % ("train", i)
            source_positive_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label = '有關'))
        file1=open("data/toy_consist.json",'r',encoding='utf-8')
        toy = []
        for line in file1.readlines():
            dic = json.loads(line)
            toy.append(dic)    
        for (i, line) in enumerate(toy):           
            guid = "%s-%s" % ("train", i)
            source_positive_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=line['Title'], label = '有關'))
        lines = self._read_tsv("data/new_artist.tsv")
        artist_name = []
        for index,i in enumerate(lines):
            if index>0:
                artist_name.append(i[4])
        times = int(max(len(baby),len(sport),len(toy)))+1
        artist_name = artist_name*times
        file1=open("data/baby_consist.json",'r',encoding='utf-8')
        baby = []
        for line in file1.readlines():
            dic = json.loads(line)
            baby.append(dic)    
        for (i, line) in enumerate(baby):           
            guid = "%s-%s" % ("train", i)
            source_negative_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=artist_name[i], label = '無關'))
        file1=open("data/sport_consist.json",'r',encoding='utf-8')
        sport = []
        for line in file1.readlines():
            dic = json.loads(line)
            sport.append(dic)    
        sport_name = rotate(sport_name,10)
        for (i, line) in enumerate(sport):           
            guid = "%s-%s" % ("train", i)
            source_negative_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=artist_name[i], label = '無關'))
        file1=open("data/toy_consist.json",'r',encoding='utf-8')
        toy = []
        for line in file1.readlines():
            dic = json.loads(line)
            toy.append(dic)    
        for (i, line) in enumerate(toy):           
            guid = "%s-%s" % ("train", i)
            source_negative_samples.append(InputExample(guid=guid, text_a=line['reviewText'],text_b=artist_name[i], label = '無關'))
        # KKBOX positve sample and negative sample
        lines = self._read_tsv("data/new_artist.tsv")
        times = int(int(len(lines)/2)/len(all_target))+1
        all_target = all_target*times
        random.Random(42).shuffle(all_target)
        name = []
        for index,i in enumerate(lines):
            if index>0:
                name.append(i[2])
        random.Random(42).shuffle(name)
        # artist shuffle
        name = rotate(name,10)
        # KKBOX review with Amazon product and inconsist artist
        negative_samples = all_target[:int(len(lines)/2)]+name[:int(len(lines)/2)+1]
        kkbox_examples = []
        kkbox_positve_exmaples = []
        kkbox_negative_examples = [ ]
        i = 0
        first_half = int(len(lines)/2)
        neg_multi_review = []
        neg_multi_target = []
        origin_target = []
        import json
        with open("data/artist.txt", "r",encoding="utf-8") as f_in1:
                    lines1 = (line.rstrip() for line in f_in1) 
                    lines1 = list(line for line in lines1 if line)
        lines2 = []
        for i in lines1:
            i = json.loads(i)
            lines2.append(i['artist'])
        one = 0
        two = 0
        three = 0
        four = 0
        above = 0
        # multi target replacement 
        for (index,line) in enumerate(lines):
            if index>0:
                count = 0
                tmp = []
                for index1, line1 in enumerate(lines2):
                    if line1 in line[0]:
                        count += 1
                        tmp.append(line1)
                if count>1:
                    line[0] = line[0].replace('<NE>','')
                    line[0] = line[0].replace('</NE>','')
                    neg_multi_review.append(line[0])
                    origin_target.append(line[4])
                    if line[4] in tmp:
                        tmp.remove(line[4])
                        random.Random(42).shuffle(tmp)
                    neg_multi_target.append(tmp[0])
                    if count == 2:
                        two+=1
                    if count == 3:
                        three+=1
                    if count == 4:
                        four+=1
                    if count > 4:
                        above+=1
        
        for (index,line) in enumerate(neg_multi_review):
                guid = "%s-%s" % ("sim",index)
                source_positive_samples.append(InputExample(guid = guid, text_a = line, text_b = origin_target[index], label = '有關'))            
        for (index,line) in enumerate(neg_multi_review):
                guid = "%s-%s" % ("sim",index)
                source_negative_samples.append(InputExample(guid = guid, text_a = line, text_b = neg_multi_target[index], label = '無關'))    
        # KKBOX positive sample
        for (i,line) in enumerate(lines):
            if i>0:
                i+=1
                guid = "%s-%s" % ("sim",i) 
                texta, textb = line[0], line[4]
                texta=texta.replace('<NE>','')
                texta=texta.replace('</NE>','')
                textb=textb.replace('<NE>','')
                textb=textb.replace('</NE>','')
                source_positive_samples.append(InputExample(guid = guid,text_a = texta,text_b = textb, label='有關'))
        i = 0
        # KKBOX review with Amazon product and inconsist artist
        for (index,line) in enumerate(lines):
            if index>0:
                i+=1
                guid = "%s-%s" % ("sim",i)
                texta, textb = line[0], negative_samples[index-1]
                texta=texta.replace('<NE>','')
                texta=texta.replace('</NE>','')
                textb=textb.replace('<NE>','')
                textb=textb.replace('</NE>','')
                source_negative_samples.append(InputExample(guid = guid, text_a = texta, text_b = textb, label = '無關'))

        random.Random(42).shuffle(source_positive_samples)
        random.Random(42).shuffle(source_negative_samples)
        return source_positive_samples, source_negative_samples

    def get_test_examples(self, data_dir):
        """See base class."""
        # None use
        examples1= []
        random.Random(42).shuffle(examples1)
        return examples1

    def get_labels(self):
        """See base class."""
        return ["有關","無關"]
    
class AugAppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        with open("nucleartransback/japanese.txt", "r") as f_in1:
            lines1 = (line.rstrip() for line in f_in1) 
            lines1 = list(line for line in lines1 if line)
        with open("nucleartransback/korea.txt", "r") as f_in2:
            lines2 = (line.rstrip() for line in f_in2) 
            lines2 = list(line for line in lines2 if line)
        with open ("nucleartransback/spanish.txt",'r') as f_in3:
            lines3=(line.rstrip() for line in f_in3)
            lines3=list(line for line in lines3 if line)
        lines=lines1+lines2+lines3
        examples = []
        i = 0 
        for (i,line) in enumerate(lines):
            i += 1
            guid = "%s-%s" % ("aug", i)
            texta=line
            examples.append(InputExample(guid=guid, text_a=texta,label="中立"))
        random.Random(42).shuffle(examples)
        return examples



appreview_processors = {
    "appreview": AppReviewProcessor,
    "unsup":  UnsupAppReviewPretrainProcessor,
    "aug": AugAppReviewPretrainProcessor,
    "sim": SimilarProcessor
    }

appreview_output_modes = {
    "appreview": "classification",
}

appreview_tasks_num_labels = {
    "appreview": NUM_CLASSES,
}

