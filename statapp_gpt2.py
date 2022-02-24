# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:49:20 2022

@author: pipaud
"""

#chargement modeles et bibli utiles
from transformers import GPT2Tokenizer,GPT2Model,GPT2LMHeadModel
import torch
from tqdm import tqdm
from math import exp

device = "cpu"
model_id = "gpt2-large"

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
max_length = model.config.n_positions

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            

def perplexity(file,limit):
    nll=0
    count=0
    total_len=0
    with open(file,encoding='utf-8') as f:
        for line in tqdm(f):
            if count>limit: # limit :  ligne à partir de laquelle commence le test set
                encodings = tokenizer(line,return_tensors='pt') # on encode la ligne
                input_ids = encodings.input_ids[:,:].to(device)
                target_ids = input_ids.clone()
                leng=encodings.input_ids.size(1) # on recupere sa taille
                target_ids[:, :-leng] = -100
                with torch.no_grad():
                        outputs = model(input_ids,labels=target_ids) #on recupere output
                        neg_log_likelihood = outputs[0]  #loss
                n=[neg_log_likelihood]
                nll+=torch.stack(n).sum() # on ajoute au compte nll 
                total_len+=leng
                

            count+=1
    return exp(nll/total_len) #on renvoie l'exp de la nll normalisée par la taille      
            

#encodings = tokenizer('Testing this.',return_tensors='pt')

#input_ids = encodings.input_ids[:,:].to(device)
#target_ids = input_ids.clone()


#with torch.no_grad():
 #   outputs = model(input_ids)
  #  neg_log_likelihood = outputs[0] 


#nll=[]
#nll.append(neg_log_likelihood)
            