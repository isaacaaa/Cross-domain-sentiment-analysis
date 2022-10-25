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
""" Finetuning multi-lingual models on XNLI (Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""


import argparse
import glob
import logging
import os
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from sklearn.metrics import confusion_matrix
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import *
from consist_processor_mtl_multi_neg import appreview_processors as processors
from consist_processor_mtl_multi_neg import appreview_output_modes as output_modes
from sklearn import metrics
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return basic_metrics(labels, preds)
def basic_metrics(y_true, y_pred):
    return {'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
            'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
            'Macro-F1': metrics.f1_score(y_true, y_pred, average='macro'),
            'Micro-F1': metrics.f1_score(y_true, y_pred, average='micro'),
            'ConfMat': confusion_matrix(y_true, y_pred)}
def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, range(len(processors['appreview']().get_labels())))
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def load_and_cache_examples(args, task, tokenizer):
    if task=='appreview':
        processor = processors['appreview']()
        unsupprocessor = processors['unsup']()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    unsupexamples= (
        unsupprocessor.get_train_examples(args.data_dir)
    )
    unsupfeatures = convert_examples_to_features(
        unsupexamples, tokenizer, max_length=args.max_seq_length,label_list= label_list, output_mode=output_mode,
    )
    unsup_all_input_ids = torch.tensor([f.input_ids for f in unsupfeatures], dtype=torch.long)
    unsup_all_attention_mask = torch.tensor([f.attention_mask for f in unsupfeatures], dtype=torch.long)
    unsup_all_token_type_ids = torch.tensor([f.token_type_ids for f in unsupfeatures], dtype=torch.long)
    if output_mode == "classification":
        unsup_all_labels = torch.tensor([f.label for f in unsupfeatures], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")
    unsupdataset = TensorDataset(unsup_all_input_ids, unsup_all_attention_mask, unsup_all_token_type_ids, unsup_all_labels)
    return unsupdataset

def evaluate(args, model1,model2,model3, tokenizer1,tokenizer2,tokenizer3, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # Using source domains tokenizer
        dataset4 = load_and_cache_examples(args, eval_task, tokenizer1)
        eval_dataset1=dataset4
        dataset4 = load_and_cache_examples(args, eval_task, tokenizer2)
        eval_dataset2=dataset4
        dataset4 = load_and_cache_examples(args, eval_task, tokenizer3)
        eval_dataset3=dataset4
        
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler1 = SequentialSampler(eval_dataset1)
        eval_dataloader1 = DataLoader(eval_dataset1, sampler=eval_sampler1, batch_size=args.eval_batch_size)
        eval_sampler2 = SequentialSampler(eval_dataset2)
        eval_dataloader2 = DataLoader(eval_dataset2, sampler=eval_sampler2, batch_size=args.eval_batch_size)
        eval_sampler3 = SequentialSampler(eval_dataset3)
        eval_dataloader3 = DataLoader(eval_dataset3, sampler=eval_sampler3, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset1))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds1 = None
        preds2=None
        preds3=None
        out_label_ids1 = None
        out_label_ids2 =None
        out_label_ids3=None
        for batch1,batch2,batch3 in tqdm(zip(eval_dataloader1,eval_dataloader2,eval_dataloader3), desc="Evaluating",total=len(eval_dataloader1)):
            # model1.eval()
            # using source domain models to evaluation
            model1.eval()
            model2.eval()
            model3.eval()
            batch1 = tuple(t.to(args.device) for t in batch1)
            batch2 = tuple(t.to(args.device) for t in batch2)
            batch3 = tuple(t.to(args.device) for t in batch3)

            with torch.no_grad():
                inputs1 = {"input_ids": batch1[0], "attention_mask": batch1[1], "labels": batch1[3]}
                if args.model_type != "distilbert":
                    inputs1["token_type_ids"] = (
                        batch1[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                inputs2 = {"input_ids": batch2[0], "attention_mask": batch2[1], "labels": batch2[3]}
                if args.model_type != "distilbert":
                    inputs2["token_type_ids"] = (
                        batch2[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                inputs3 = {"input_ids": batch3[0], "attention_mask": batch3[1], "labels": batch3[3]}
                if args.model_type != "distilbert":
                    inputs3["token_type_ids"] = (
                        batch3[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs1 = model1(**inputs1)
                outputs2=model2(**inputs2)
                outputs3=model3(**inputs3)
            
                tmp_eval_loss1, logits1 = outputs1[:2]
                tmp_eval_loss2, logits2 = outputs2[:2]
                tmp_eval_loss3, logits3 = outputs3[:2]

                eval_loss += (tmp_eval_loss1.mean().item()+tmp_eval_loss2.mean().item()+tmp_eval_loss3.mean().item())//3
            nb_eval_steps += 1
            if preds1 is None:
                preds1 = logits1.detach().cpu().numpy()
                out_label_ids1 = inputs1["labels"].detach().cpu().numpy()
            else:
                preds1 = np.append(preds1, logits1.detach().cpu().numpy(), axis=0)
                out_label_ids1 = np.append(out_label_ids1, inputs1["labels"].detach().cpu().numpy(), axis=0)
            if preds2 is None:
                preds2 = logits2.detach().cpu().numpy()
                out_label_ids2 = inputs2["labels"].detach().cpu().numpy()
            else:
                preds2 = np.append(preds2, logits2.detach().cpu().numpy(), axis=0)
                out_label_ids2 = np.append(out_label_ids2, inputs2["labels"].detach().cpu().numpy(), axis=0)
            if preds3 is None:
                preds3 = logits3.detach().cpu().numpy()
                out_label_ids3 = inputs3["labels"].detach().cpu().numpy()
            else:
                preds3 = np.append(preds3, logits3.detach().cpu().numpy(), axis=0)
                out_label_ids3 = np.append(out_label_ids3, inputs3["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds1 = np.argmax(preds1, axis=1)
            preds2 = np.argmax(preds2, axis=1)
            preds3 = np.argmax(preds3, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        preds_list1=preds1.tolist()
        preds_list2=preds2.tolist()
        preds_list3=preds3.tolist()
        from collections import Counter
        
        final_pred=[]
        for index , i in enumerate(preds_list1):    
            preds=[]    
            preds.extend([i,preds_list2[index],preds_list3[index]])
            
            c = Counter(preds)
            # prediction's majority
            value, count = c.most_common()[0]
            # get pseudo label
            final_pred.append(value)
        print(len(final_pred))
        final_pred=np.array(final_pred)
        print(len(final_pred))
        print(final_pred)
        np.save('new_new_unlabelled_pseudo_label.npy',final_pred)
        result = compute_metrics(eval_task, preds3, out_label_ids1)
        
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            import logging
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results
def main():
   
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='sorted_data_acl/',
        type=str,
       
        help="The input data dir. Should contain the .tsv files (or other data files) for the task. FOR TARGET DOMAIN",
    )
    
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        
        help="Model type selected in the list: "+".join(MODEL_CLASSES.keys())" ,
    )
    parser.add_argument(
        "--task_name",
        default="appreview",
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-chinese',
        type=str,
        
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--output_dir",
        default='models/baby',
        type=str,
        
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--target_domain", default='books',type=str, help="Choose target domain,remains domains are sources domain")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_unsup_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for unsupervised training.")
    parser.add_argument("--loss_weight", default=0.5, type=float, help="loss weight between supervised loss and unsupervised loss")
    parser.add_argument("--uda_confidence_thresh", default=0.45, type=float, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--uda_softmax_temp",default=0.85,type=float,help="aaa")
    parser.add_argument("--tsa",default=True,type=bool, help="whether uda tsa or not")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--source", type=str, default="", help="For distant debugging.")

    
    args = parser.parse_args()
    args.do_train=False
    args.do_eval=True
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    # args.no_cuda=True
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    import logging
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    if args.task_name=='appreview':
        processor = processors[args.task_name]()
        unsupprocessor = processors['unsup']()
        augprocessor = processors['aug']()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        #num_labels=len(label_list),
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_labels = len(label_list)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logger.info("Training/evaluation parameters %s", args)

   
   
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        '''
        source models1 baby
        '''
        tokenizer1 = tokenizer_class.from_pretrained('models/baby')
        checkpoints = ['models/baby']
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model1 = model_class.from_pretrained(checkpoint)
            model1.to(args.device)
        '''
        souurce models2 sport
        '''
        tokenizer2 = tokenizer_class.from_pretrained('models/sport')
        checkpoints = ['models/sport']
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model2 = model_class.from_pretrained(checkpoint)
            model2.to(args.device)
        '''
        source models3 toy
        '''
        tokenizer3 = tokenizer_class.from_pretrained('models/toy')
        checkpoints = ['models/toy']
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model3= model_class.from_pretrained(checkpoint)
            model3.to(args.device)
        
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        model4 = model_class.from_pretrained(checkpoint)
        model4.to(args.device)
        result = evaluate(args, model1,model2,model3, tokenizer1,tokenizer2,tokenizer3, prefix=prefix)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

    return results


if __name__ == "__main__":
    main()
    

