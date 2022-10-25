import torch.nn as nn
import json
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
# from load_data import prepare_source,prepare_evaluate,prepare_data,load_obj,save_obj
import os,sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from domain_similarity import compute_psi_for_test
# from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import argparse
import glob
import random
import logging
import os
from scipy.special import softmax
import random
from torch import optim
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
device = torch.device("cuda")
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from consist_processor_mtl_multi_neg import appreview_processors as processors
from consist_processor_mtl_multi_neg import appreview_output_modes as output_modes
from sklearn import metrics
from transformers import BertTokenizer, BertModel
from sklearn import metrics
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
 
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)
from modeling_bert_share import BertForSequenceClassification
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, range(len(processors['appreview']().get_labels())))

def basic_metrics(y_true, y_pred):
    return {'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
            'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
            'Macro-F1': metrics.f1_score(y_true, y_pred, average='macro'),
            'Micro-F1': metrics.f1_score(y_true, y_pred, average='micro'),
            'ConfMat': confusion_matrix(y_true, y_pred)}

# xavier initialization to initialize source domain embedding
def init_w(embedding_dim):
    w = torch.Tensor(embedding_dim,1)

    return nn.Parameter(nn.init.xavier_uniform_(w).to(device)) # sigmoid gain=1
def get_source_instance(args,data_dir,evaluate):
    processor=processors['appreview']()
    unsup_processor=processors['unsup']()
    if evaluate==True:
        examples1,examples2,examples3,examples4 = (
            processor.get_train_examples(data_dir)
        )
        t_examples4 = (
            processor.get_test_examples(data_dir)
        )
        t_kitchen,t_kitchen_label=get_bert_embedding(args,t_examples4)
        return t_kitchen,t_kitchen_label
    else:
        examples1,examples2,examples3,examples4 = (
            processor.get_train_examples(data_dir)
            )
        baby,b_label=get_bert_embedding(args,examples1)
        sport,s_label=get_bert_embedding(args,examples2)     
        toy,t_label=get_bert_embedding(args,examples3)
        return baby,b_label,sport,s_label,toy,t_label
#get bert embedding
def get_bert_embedding(args,examples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese',return_dict=True)
    label_list=['正評','中立','負評']
    features1 = convert_examples_to_features(
        examples, tokenizer, max_length=args.max_seq_length , label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    books_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    train_sampler = SequentialSampler(books_dataset)
    train_dataloader = DataLoader(books_dataset, sampler=train_sampler, batch_size=16,drop_last=True)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
    book=[]
    label=[]
    for step, batch in enumerate(epoch_iterator):
            model.to(device)
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3]}
            label.append(inputs["labels"])
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2]}
            outputs = model(**inputs)
            book.append(outputs.pooler_output.detach().cpu())
    return book,label
    
def reshape(book):
    for i in book:
        continue
    book= torch.stack(book, 0)
    x=book.shape[0]
    y=book.shape[1]#16=number of instance
    z=book.shape[2]#768=embedding_dim
    book=book.view(x*y,768)
    return book

def reshape_label(b_label):
    b_label=torch.stack(b_label,0)
    x=b_label.shape[0]
    y=b_label.shape[1]
    b_label=b_label.view(-1,x*y)
    b_label=b_label.squeeze(0)
    return b_label

def cos_sim(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) if (np.linalg.norm(a)*np.linalg.norm(b))!=0 else 0
    return cos_sim
def compute_centriod(instances):
    a = np.array(instances)
    print(a.shape)
    print (np.mean(a,axis=0).shape)
    return np.mean(a,axis=0)

def unlabel_sim(tgt_un):
    filter_u_cat=[]
    c_t = compute_centriod(tgt_un)
    computed_tgt_sim = [cos_sim(x,c_t) for x in tgt_un]
    print('simsimsimsismismismismism')
    print(computed_tgt_sim)
    topk=sorted(range(len(computed_tgt_sim)),key=lambda i:computed_tgt_sim[i])[-5000:]
    for i in topk:
        filter_u_cat.append(tgt_un[i])
    return filter_u_cat,topk

# class for attention based negative samples selection
class NegativeSamplesAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(NegativeSamplesAttention, self).__init__()
        # self.y = y
        self.phi_srcs = nn.ParameterList([init_w(embedding_dim) for i in range(1)])

        self.bias = nn.Parameter(torch.Tensor([0]).to(device))

        self.sigmoid = nn.Sigmoid()

    def forward(self,negative,positive):
        for phi_src in self.phi_srcs:
            m = nn.Softmax(dim=1)
            # calculate cosine similarity for negative samples embedding and positive sample embedding
            cos = nn.CosineSimilarity(dim = 0, eps = 1e-6)
            sim = cos(negative, positive)
            # calclate attention
            attention = F.softmax(torch.tanh(phi_src*sim+self.bias))
            output = torch.mm(attention, negative.T)
            attention_cls = nn.Linear(768,2).cuda()
            # new negative sample's embedding after attention transform
            output = attention_cls(output.T).view(-1,2)
        return output


class DomainAttention(nn.Module):
    # embedding_dim: embedding dimensionality
    # hidden_dim: hidden layer dimensionality
    # source_size: number of source domains
    # hidden_dim, batch_size, label_size, ,num_instances=1600
    def __init__(self, embedding_dim, source_size, y,label_size=3):
        super(DomainAttention, self).__init__()
        # bias
        self.bias = nn.Parameter(torch.Tensor([0]).to(device))
        # source phi, multi sources, just like python list
        self.phi_srcs = nn.ParameterList([init_w(embedding_dim) for i in range(source_size)])
        # labels for source domain labeled data
        self.y = y
        self.label_size = label_size
        self.sigmoid = nn.Sigmoid()
        self.source_size = source_size
        
    def forward(self, batch_x,cat):
        all_y_hat=[]
        # x:target instance
        x=batch_x.to(device)
        # y: source domain labeled data's label
        y = self.y.view(-1,len(self.y)).to(device)
        con=[]
        psisum=0
        # instance level attention
        for i in cat:
            i=torch.unsqueeze(i,0)
            tmp=torch.mm(i,x.T)
            con.append(tmp)
        con= torch.stack(con, 0)
        shapex=con.shape[0]
        con= con.view(shapex, -1)
        con=F.normalize(con,p=4,dim=0)
        m=nn.Softmax(dim=0)
        con=m(con).to(device)
        psi_splits = torch.chunk(con,self.source_size,dim=0)
        y = torch.chunk(y,self.source_size,dim=1)
        theta_splits = []
        sum_src = 0.0
        # domain level attention
        theta_splits1 = []
        for phi_src in self.phi_srcs:
            x=x.to(device)
            temp = torch.exp(torch.mm(x,phi_src))
            theta_splits.append(temp)
            sum_src+=temp
        sum_matrix = 0.0
        count = 0
        # concat instance level attention and domain level attention to get prediction for target unlabeled instance
        for theta,psi_split,y_split in zip(theta_splits,psi_splits,y):
            count += 1
            theta_matrix = theta/sum_src
            y_split=y_split.to(device).float()
            psi_split=psi_split.to(device).float()
            theta=theta.to(device).float()
            a=y_split.T*psi_split
            temp=torch.mul(a,theta.T)
            sum_matrix += torch.sum(temp,0)
        sum_matrix = sum_matrix  + self.bias
        y_hat = sum_matrix
        all_y_hat.append(y_hat)
        return all_y_hat

def train(args,model,tokenizer,labelled_kitchen,u_kitchen,y_train,cat,loss_function,optimizer,i,rescale, pos_samples, neg_examples):
    embedding_set=[]
    embedding_label_set=[]
    label_list=['正評','中立','負評']
    sim_label_list=['有關','無關']
    # call the attention based negative samples selection model
    negative_attention = NegativeSamplesAttention(embedding_dim = 768)
    model.classifier4 = negative_attention
    model.train()
    #labelled feature
    sup_features1 = convert_examples_to_features(
        labelled_kitchen, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    sup_all_input_ids1= torch.tensor([f.input_ids for f in sup_features1], dtype=torch.long)
    sup_all_attention_mask1 = torch.tensor([f.attention_mask for f in sup_features1], dtype=torch.long)
    sup_all_token_type_ids1 = torch.tensor([f.token_type_ids for f in sup_features1], dtype=torch.long)
    sup_all_labels1 = torch.tensor([f.label for f in sup_features1], dtype=torch.long)
    #unlabelled feature
    features1 = convert_examples_to_features(
        u_kitchen, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    #negative relevant feature#
    neg_features = convert_examples_to_features(
        neg_examples, tokenizer, max_length=256, label_list=sim_label_list, output_mode='classification',
    )
    neg_all_input_ids= torch.tensor([f.input_ids for f in neg_features], dtype=torch.long)
    neg_all_attention_mask = torch.tensor([f.attention_mask for f in neg_features], dtype=torch.long)
    neg_all_token_type_ids = torch.tensor([f.token_type_ids for f in neg_features], dtype=torch.long)
    neg_all_labels = torch.tensor([f.label for f in neg_features], dtype=torch.long)
    #positive relevant feature#
    pos_features = convert_examples_to_features(
        pos_samples, tokenizer, max_length=256, label_list=sim_label_list, output_mode='classification',
    )
    pos_all_input_ids= torch.tensor([f.input_ids for f in pos_features], dtype=torch.long) 
    pos_all_attention_mask = torch.tensor([f.attention_mask for f in pos_features], dtype=torch.long)
    pos_all_token_type_ids = torch.tensor([f.token_type_ids for f in pos_features], dtype=torch.long)
    pos_all_labels = torch.tensor([f.label for f in pos_features], dtype=torch.long)
    all_labels1= y_train
    batch_size=16
    def slice(feature):
        concate=[]
        for index,x,y in enumerate(zip(sup_feature,unsup_feature)):
            concate.append()
    #dataset
    label_dataset = TensorDataset(sup_all_input_ids1, sup_all_attention_mask1, sup_all_token_type_ids1, sup_all_labels1)
    unlabel_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    neg_dataset = TensorDataset(neg_all_input_ids, neg_all_attention_mask, neg_all_token_type_ids, neg_all_labels)
    pos_dataset = TensorDataset(pos_all_input_ids, pos_all_attention_mask, pos_all_token_type_ids, pos_all_labels)
    #sampler
    sup_sampler = RandomSampler(label_dataset)
    unsup_sampler = RandomSampler(unlabel_dataset)
    neg_sampler = SequentialSampler(neg_dataset)
    pos_sampler = SequentialSampler(pos_dataset)
    #dataloader
    sup_dataloader = DataLoader(label_dataset, sampler=sup_sampler, batch_size=args.per_gpu_suptrain_batch_size)
    sup_iterator = tqdm(sup_dataloader, desc="Iteration", disable=False)
    unsup_dataloader = DataLoader(unlabel_dataset, sampler=unsup_sampler, batch_size=args.per_gpu_unsuptrain_batch_size)
    unsup_iterator = tqdm(unsup_dataloader, desc="Iteration", disable=False)
    neg_dataloader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=args.per_gpu_simtrain_batch_size)
    neg_iterator = tqdm(neg_dataloader, desc="Iteration", disable=False)
    pos_dataloader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=args.per_gpu_simtrain_batch_size)
    pos_iterator = tqdm(pos_dataloader, desc="Iteration", disable=False)
    
    avg_loss = 0.0
    sup_list = []
    neg_list = []
    pos_list = []
    pos_irr_list = []
    # prepare batch training data
    # classifier1 for supervised training
    # classifier2 for transfer learning
    # classifier3 and classifier4 for relevant task detection
    # target labeled data
    for step ,batch in enumerate(sup_iterator):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier1'}
        sup_list.append(inputs)
    # relevant task negative samples
    for step ,batch in enumerate(neg_iterator):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier4'}
        neg_list.append(inputs)
    # relevant task positive samples
    for step ,batch in enumerate(pos_iterator):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier3'}
        pos_list.append(inputs)
    # target unlabeled data
    for step, batch in enumerate(unsup_iterator):
        output_list=[]
        model.zero_grad()    
        model.to(device)
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier2'}
        #model training
        mod = step % len(sup_iterator)
        unsup_outputs = model(**inputs)
        sup_outputs=model(**sup_list[mod])
        mod1 = step % len(neg_iterator)
        mod2 = step % len(pos_iterator)
        neg_outputs=model(**neg_list[mod1])
        pos_outputs=model(**pos_list[mod2])
        negative_attention_outputs = model.classifier4(neg_outputs[0], pos_outputs[1] )
        relevant_truth = neg_list[mod1]['labels'].view(-1)
        loss_fuc = nn.CrossEntropyLoss()
        relevant_loss = loss_fuc(negative_attention_outputs, relevant_truth.to(device))
        truth=batch[3]
        optimizer.zero_grad()
        pred=torch.unsqueeze(unsup_outputs['logits'][0],1)
        linear=nn.Linear(1,3).cuda()
        output1=linear(pred)
        unsup_loss = loss_function(output1, truth.to(device))
        sup_loss=sup_outputs[0]
        pos_loss=pos_outputs[0]
        sim_loss = (pos_loss + relevant_loss)/2
        # final loss = supervised training loss + args.weights*unsupervised training loss+args.sim_weights*relevant task loss
        loss=sup_loss+args.weights*unsup_loss+args.sim_weights*sim_loss
        avg_loss+=loss
        loss.backward()#retain_graph=True
        optimizer.step()
    return avg_loss

            
def evaluate_epoch( args,model,tokenizer,test_examples, cat, loss_function,classifier):
    #model evaluation
    model.eval()
    label_list=['正評','中立','負評']
    # get testing data's features
    features1 = convert_examples_to_features(
        test_examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=16)
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    truth_res = []
    pred_list=[]
    preds_1=None
    out_label_ids=None
    ori_pred=[]
    print('Start evaluating!!!')
    for step, batch in enumerate(epoch_iterator):
        output_list=[]
        model.to(device)
        batch = tuple(t.to(device) for t in batch)
        # use transfer learning classifier to predict
        if classifier=='classifier2':
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],'cat':cat,'return_dict':True,'classifier':classifier}
                outputs = model(**inputs)
                truth=batch[3].float().detach()
                truth_res+=truth.tolist()
                pred=torch.unsqueeze(outputs['logits'][0],1) 
                linear=nn.Linear(1,3).cuda()
                output1=linear(pred)
                softmax=nn.Softmax(dim=1)
                output2=softmax(output1)
                softmax_preds= np.argmax(output2.cpu(), axis=1)
                preds_1_list=softmax_preds.tolist()
                pred_list=pred_list+preds_1_list        
                
        # use supervised training classifier to predict
        else:
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],'labels':batch[3],'cat':cat,'return_dict':True,'classifier':classifier}
                truth=batch[3]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                preds_1 = logits.detach().cpu().numpy()
                softmax_preds= np.argmax(preds_1, axis=1)
                preds_1_list=softmax_preds.tolist()
                truth_res+=truth.tolist()
                pred_list=pred_list+preds_1_list        
    
    from sklearn.metrics import accuracy_score
    # get confusion matrix
    results =basic_metrics(truth_res,pred_list)
    print('eval_results:',results)
    return results,truth_res,pred_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default='output/',
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
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_suptrain_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_unsuptrain_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_simtrain_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--weights" , default=0.3 , type=float , help="weight between supervised training loss and transfer learning loss"
            )
    parser.add_argument("--sim_weights" , default=0.3 , type=float , help="weight between supervised training loss and transfer learning loss"
            )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
            "--num_train_epochs", default=11, type=int, help="Total number of training epochs to perform."
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
    parser.add_argument("--num_unlabelled", type=int, default=5000, help="number of unlabelled data in target domain")
    parser.add_argument("--eval_classifier", type=str, default='classifier1', help="evaluate classifier1: supervised learning or classsifier2:transfer learning")
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
    args = parser.parse_args()
    baby,b_label,sport,s_label,toy,t_label =get_source_instance(args,"data",False)
    # get soruce domain labeled data's label embedding
    b_label=reshape_label(b_label)
    s_label=reshape_label(s_label)
    t_label=reshape_label(t_label)
    cat_label=torch.cat((b_label,s_label),0)
    cat_label=torch.cat((cat_label,t_label),0)
    baby=reshape(baby)
    sport=reshape(sport)
    toy=reshape(toy)
    # get source domain labeled data embedding
    cat=torch.cat((baby,sport),0)
    cat=torch.cat((cat,toy),0)
    cat=cat.to(device)
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained('bert-base-chinese')
    tokenizer = tokenizer_class.from_pretrained('bert-base-chinese')
    model = model_class.from_pretrained('bert-base-chinese')
    model.classifier2=DomainAttention(embedding_dim = 768, 
                            source_size = 3,
                            y =cat_label)
    model.classifier4 = NegativeSamplesAttention (embedding_dim = 768)
    model.num_labels=3
    # optimizer
    optimizer = optim.Adam([
        {"params":model.bert.parameters(),"lr":args.learning_rate},
        {"params":model.classifier1.parameters(),"lr":1e-5},
        {"params":model.classifier2.parameters(),"lr":1e-5},
        {"params":model.classifier3.parameters(),"lr":1e-5},
        {"params":model.classifier4.parameters(),"lr":1e-5}],
        lr = 1e-5
        )
    
    bce_loss_function=nn.BCELoss(reduction='mean')
    loss_function=nn.CrossEntropyLoss()
    best_f1=0.0
    x=args.num_train_epochs-1
    result_list=[]
    y_star=np.load('new_new_unlabelled_pseudo_label.npy')
    y_star=y_star[:args.num_unlabelled]
    y_star=torch.from_numpy(y_star)
    r = 0.5*0.6
    a0 = 1.0
    import os.path
    from os import path
    if path.exists('proposed/target_results.txt')==True:
        os.remove('proposed/target_results.txt')
    if path.exists('proposed/prediction.tsv')==True:
        os.remove('proposed/prediction.tsv')
    unsup_processor=processors['unsup']()
    sim_processor = processors['sim']()
    unsup_examples4 = (
            unsup_processor.get_train_examples('sss')
        )
    processor=processors['appreview']()
    examples1,examples2,examples3,examples4 = (
            processor.get_train_examples('sss')
        )
    t_examples4,text = (
            processor.get_test_examples('sss')
        )
    if args.do_train:
        for i in range(args.num_train_epochs):
            positive_samples, negative_samples = (
                    sim_processor.get_train_examples('sss')
                    )
            sim_test_exampls = (
                    sim_processor.get_test_examples('sss')
                    )
            #training
            loss=train(args,model,tokenizer,examples4,unsup_examples4[:args.num_unlabelled],y_star,cat,loss_function,optimizer,i,0,positive_samples,negative_samples)
            #testing
            results,truth,pred=evaluate_epoch(args,model,tokenizer,t_examples4,cat,loss_function,'classifier1')
            result_list.append(results)
            if results['Macro-F1']>best_f1:
                best_f1=results['Macro-F1']
                print('best_epoch:',i+1)
                print('best_epoch_F1:',best_f1)
                import os
                import shutil
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                else: 

                    shutil.rmtree(args.output_dir)
                    os.makedirs(args.output_dir)
            
                torch.save(model,args.output_dir+'/target_model')
                tokenizer.save_pretrained(args.output_dir)
            if i ==x:
                with open('proposed/target_results.txt','w') as f:
                    f.write(str(best_f1))
                    f.write('\n')
                    f.write(str(result_list))
                import csv
                with open('proposed/prediction.tsv','w')as f:
                    writer=csv.writer(f,delimiter='\t')
                    writer.writerow(['text','truth','pred'])
                    writer.writerows(zip(text,truth,pred))
        print('model_best_performance:',best_f1)
    if args.do_eval:
        eval_model = torch.load(args.output_dir+'/target_model')
        tokenizer = tokenizer_class.from_pretrained('proposed/')
        results,truth,pred=evaluate_epoch(args,eval_model,tokenizer,t_examples4,cat,loss_function,'classifier1')
        print('eval_performance:',results)
    return results

if __name__ == "__main__":
    main()
