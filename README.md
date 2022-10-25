# A Teacher-Student Approach to Cross-Domain Transfer Learning with Multi-level Attention (WI-IAT 2022)
This is the implementation of the paper [A Teacher-Student Approach to Cross-Domain Transfer Learning with Multi-level Attention](https://dl.acm.org/doi/abs/10.1145/3486622.3494009) .

## Model architecture
![image](https://github.com/kklab-com/cross-domain-sentiment-analysis/blob/master/transfer%20learning%20architecture.png)

## Model introduction and operation steps: 
https://docs.google.com/presentation/d/1nE51EONlOcYwqN-rqkhXBqgWWdFWAQZOZbEq_pSxo7E/edit#slide=id.p
<!-- 
## Weekly report:
https://docs.google.com/presentation/d/1V1n9mJ-qUIKCv9-1Z7h1cVSJ6jPPvBPZ7hH42bu3ahQ/edit#slide=id.g9f9e4d4e94_1_0
https://docs.google.com/presentation/d/1zPLn2K_5FxgEP8SrbvLVvBF7I1Sp7boyP5Qxi1wfNq8/edit#slide=id.gdadffaed61_0_12 -->

## Proposal Defense:
https://docs.google.com/presentation/d/1XEMM0AV5clQdjW1wEFO3Sf1hK0PFwJ_GqcIKXqNO-2c/edit#slide=id.p

## Training dataset and fine-tuned models
[Amazon Chinese Dataset](https://drive.google.com/drive/folders/14nvjHSRDwNUqdJqlHJG_XFYwTFmoBqJO?usp=sharing)

[Source models](https://drive.google.com/drive/folders/1NdE0b-Yvk1SetUHR-8Jwi488O7VxlqAi?usp=sharing)

[Target model](https://drive.google.com/drive/folders/1OukLJ2F2TSXJo9ZGkUDTD8HB1rSsqzTj?usp=sharing)


## Environment Setup
```
virtualenv -p python3 venv
```
```
source ./venv/bin/activate
```
```
pip3 install -r requirements.txt
```


<!-- -- final_cross_domin_uda_mtl.py: our model

-- run.sh: run our model

-- consist_processor_mtl_multi_neg.py: data processor for final_cross_domin_uda_mtl.py (with multi target)

-- majority_vote.py: majority vote for unlabelled target instance using fine-tuning source domain classifier 

-- new_new_unlabelled_psuedo_label.npy: the result after majority vote

-- data link: https://drive.google.com/drive/folders/14nvjHSRDwNUqdJqlHJG_XFYwTFmoBqJO?usp=sharing
              
              --baby_consist.json: Amazon baby domain product review (labeled)
              
              --sport.json: Amazon sport domain product review (labeled)
              
              --toy.json: Amazon toy domain product review (labeled)
              
              --new_artist.tsv: KKBOX artist reviews (labeled)
              
              --artist.txt: the list of artist
              
              --kkbox_negative_review: PTT negative reviews (labeled)
              
              --ptt.tsv (artist unlabeled review)

-- source domain model link: https://drive.google.com/drive/folders/1NdE0b-Yvk1SetUHR-8Jwi488O7VxlqAi?usp=sharing
 
-- cross domain output files will be in "proposed", link: https://drive.google.com/drive/folders/1OukLJ2F2TSXJo9ZGkUDTD8HB1rSsqzTj?usp=sharing

--bert.py: bert baseline model

--run.sh: run bert baseline -->

<!-- --cross_processor.py: data processor for bert.py

--bert baseline output file will be in "output" -->





## Usage

Run bert baseline model
```
sh run_bert_base.sh
```
Run paper's model
```
sh run.sh
```
