
import numpy as np 
from tqdm import tqdm
import json
def cut_strings_term(s):
    s=s.replace('.', ' .')
    s=s.replace('?', ' ?')
    s=s.replace('!',' !')
    s=s.replace('_','')
    s=s.replace('  !',' !')
    s=s.replace('  ?',' ?')
    s=s.replace('  .',' .')
    return s.split(' ')


def begin_end(N,L):
    for k in range(N):
        M=['<s>' for k in range(N)]
    n=len(L)
    c=0
    for k in range(n):
        if L[k]=='.' or L[k]=='?' or L[k]=='!':
            if k<n-1:
                M+=L[c:k]
                c=k+1
                M.append(L[k])
                M.append('<s/>')
                for k in range(N):
                    M.append('<s>')
            elif k==n-1:
                M+=L[c+1:k+1]
                c=k
                M.append('<s/>')              
    if c<(n-1):
        M+=L[c:]
    if not(M[-1]=='<s/>'):
        M.append('<s/>')   
    return M


def treat_txt(N,string):
    L=cut_strings_term(string)
    return begin_end(N,L)


def assemble(L):
    s=''
    n=len(L)
    for k in range(n):
        s+=L[k]
        s+=' '
    return s 

def make_gstr(L):
    l=len(L)
    s='_'+str(L[0])
    for k in range(1,l):
        s+='_'+str(L[k])
    return s+'_'

## transforme une liste de mots ['Je','vais','à','la','plage']
#en '_Je_vais_à_la_plage_'

def cut_gstr(s):
    L=[]
    l=len(s)
    for k in range(l):
        if s[k]=='_':
            L.append(k)
    m=len(L)
    M  = [s[L[k]+1:L[k+1]] for k in range(m-1)]
    return M

# transformation inverse à la précédente (utile pour recupérer le n-1 gram quand on a le ngram sous la forme '__blabla__')
    

def count_grams(N,txt,dico1,dico2):
    text=np.array(treat_txt(N,txt))
    n=len(text)
    # N + k < n 
    for k in range(n):
        if N+k < n:
            t1=make_gstr(text[k:N+k])
            if t1 in dico1:
                dico1[t1]+=1
            else:
                dico1[t1]=1
        if N+k-1<n:
            t2=make_gstr(text[k:N+k-1])
            if t2 in dico2:
                dico2[t2]+=1
            else:
                dico2[t2]=1
    #return dico1,dico2
    

#count_words = 0

#with open(file,encoding='utf-8') as f:
 #   for line in f:
  #      L = cut_strings_term(line)
   #     count_words += len(L)
    

#count_words
#Out[5]: 60075583

#0.8*count_words
#Out[6]: 48060466.400000006
 

# count_words = 0

#with open(file,encoding='utf-8') as f:
 #   for line in f:
  #      L = cut_strings_term(line)
   #     count_words+=len(L)
    #    if not(count_words>train_limit):
     #       count_lines+=1
            

#count_lines
#Out[15]: 796351

train_limit_lines = 796351

def grams_lbl_k_train(N,file,k,dico1,dico2,limit):
    with open(file,'r',encoding='utf-8') as f:
        text=''
        c=0
        for line in tqdm(f):
            if (c>=k*50000) and (c<(k+1)*50000) and c<train_limit_lines:
                    if not type(text)==str:
                        line = ' '.join(text)+' '+line          
                    count_grams(N,line,dico1,dico2)
                    text=np.array(treat_txt(N,line))[-N:]
            c+=1


def grams_lbl_k(N,file,k,dico1,dico2):
    with open(file,'r',encoding='utf-8') as f:
        text=''
        c=0
        for line in tqdm(f):
            if (c>=k*50000) and (c<(k+1)*50000):
                    if not type(text)==str:
                        line = ' '.join(text)+' '+line          
                    count_grams(N,line,dico1,dico2)
                    text=np.array(treat_txt(N,line))[-N:]
                    c+=1
                        
            else:
                c+=1
                
#c nb lignes oscar

#Out[14]: 1000061

import gc

L = [15,16,17,18,19,20]

def grams_lbl(N,file):
    dico1={}
    dico2={}
    k=0
    for k in L:
        grams_lbl_k(N,file,k,dico1,dico2)
        gc.collect()
        
    return dico1,dico2

def grams_lbl_train(N,file,limit):
    dico1={}
    dico2={}
    k=0
    for k in range(5):
        grams_lbl_k_train(N,file,k,dico1,dico2,train_limit_lines)
        gc.collect()
        
    return dico1,dico2




def est_prob(dico1,dico2):
    probas={} #dictionnaire de dictionnaire
    for couple1 in tqdm(dico1.items()):
        Ngram,occurences=couple1 #on récupère chq Ngram et son nb d'occurences
        L = cut_gstr(Ngram)
        L_= L[:-1]
        ngramm = L[-1]
        n_1 = '_'+'_'.join(L_)+'_'
        #n-1 gram corresponcant au Ngram en cours
        if not n_1 in probas: #si ce n-1 gram n'est pas dans probas
            dic_n1 = {} # on crée un dico 
            dic_n1[ngramm]=occurences/dico2[n_1] 
            # on def la proba du dernier mot du N gram
            # sachant le N-1 gram
            
            probas[n_1]=dic_n1 #on associe le n-1 gram au dico
            # contenant les mots suivant le n-1 et les probas associ&s
        else: # sinon
        #on recup le dico déjà crée
            dico_n1 = probas[n_1]
            # on lui ajoute une valeur pr le n gram en cours
            
            dico_n1[ngramm]=occurences/dico2[n_1]
            probas[n_1]=dico_n1
            
    return probas


def pred2(N,gen,probas):    
    if not(gen==''):
        gen = cut_strings_term(gen)
        gen = ['<s>' for k in range(N)] + gen
    else:
        gen = ['<s>' for k in range(N)]
    gen = np.array(gen)
    k=0
    while not(gen[-1]=='<s/>') and k<50:
        seq = make_gstr(gen[-N+1:])
        dico=probas[seq]
        L=sorted(dico.items(), key=lambda mot_proba: mot_proba[1])
        mot = L[-1][0]
        if mot=='<s>':
            mot = L[-2][0]
        gen= np.append(gen,mot)
        k+=1
        
    return assemble(gen)

def keysvalues(dico):
    keys=[]
    values=[]
    for couple in dico.items():
        mot,proba=couple
        keys.append(mot)
        values.append(proba)
    return keys,values

def softmax(prob,t):
    prob_sm = np.array([exp(-probs/t) for probs in prob])
    prob_sm= prob_sm/prob_sm.sum()
    return prob_sm
        

# pour sampling "pur", prendre temperature=0
def pred2sampling(N,gen,model,temperature):
    if not(gen==''):
        gen = cut_strings_term(gen)
        gen = ['<s>' for k in range(N)] + gen
    else:
        gen = ['<s>' for k in range(N)]
    gen = np.array(gen)
    k=0
    while not(gen[-1]=='<s/>') and k<50:
        seq = make_gstr(gen[-N+1:])
        dico=model[seq]
        mots,probas=keysvalues(dico)
        if not(temperature==0):
            probas = softmax(probas,temperature)
        L=np.random.multinomial(1,probas,size=1)
        m=np.argmax(L)
        mot = mots[m]
        gen= np.append(gen,mot)
        k+=1
    return assemble(gen)
        
        
        
            
from math import log
from math import exp

def sequence_prob(N,s,model):
    L = cut_strings_term(s)
    L=np.array(L)
    probas=0
    n=len(L)
    for k in range(n):
        if N+k<n:
            seq=make_gstr(L[k:N+k-1])
            if seq in model:
                dico = model[seq]
                next_word = L[N+k-1]
                if next_word in dico:
                    prob = dico[next_word]
                    probas =  probas + log(prob)
    return probas,n
            

def perplexity(N,s,model):
    m=len(s)
    val = sequence_prob(N,s,model)
    if val==0:
        return 'not defined'
    else:
        c = (-1/m)*(val)
        return exp(c)

def perplexity_file(N,file,model,limit):
    i=0
    probas = 0
    m=0
    with open(file,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            if i>=limit:
                prob,n=sequence_prob(N,line,model)
                probas+=prob
                m+=n
            i+=1
    probas = (-1/m)*probas
    return probas


def perplexity_split(N,file,train_prop):
    count_w = 0
    with open(file,encoding='utf-8') as f:
        for line in f:
            L = cut_strings_term(line)
            count_w += len(L)
    limit_words = train_prop * count_w
    limit_line=0
    count_w=0
    dico1,dico2 = grams_lbl_train(N,file,limit_line)
    model = est_prob(dico1,dico2)
    with open(file,encoding='utf-8') as f:
        for line in f:
            L = cut_strings_term(line)
            count_w+=len(L)
            if count_w < limit_words:
                limit_line+=1
    perplexity = perplexity_file(N,file,model,limit_line)
    return perplexity
     
# fonctionnement : 
    # on parcoure une fois le fichier pour trouver à partir de quelle ligne
    # on a atteint train_pop*100 % du fichier (on compte les mots)
# puis, on entraine sur les lignes avant cette limite
# puis on calcule la perplexité sur le reste


#perplexity_file(2,file,probas)
#1000061it [01:57, 8484.70it/s] 
#Out[36]: 45.72483452455676

"frankenstein 6187"
#perplexity_file(2,file,probas,i)
#7743it [00:00, 88980.60it/s]
#Out[19]: 13.375999785434162

#perplexity_file(3,file,probas,i)
#7743it [00:00, 97483.98it/s]
#Out[22]: 1.8332584781477466

#perplexity_file(4,file,probas,i)
#7743it [00:00, 95790.14it/s]
#Out[25]: 1.0709281496096408

#
#perplexity_file(5,file,probas,i)
#7743it [00:00, 99979.98it/s]
#Out[28]: 1.0102932779346747

#perplexity_file(10,file,probas,i)
#7743it [00:00, 140851.25it/s]
#Out[31]: 1.0
     

# OSCAR

#perplexity_split(2,file,0.8)
#1000061it [01:01, 16241.77it/s]
#1000061it [00:56, 17729.16it/s]
#1000061it [01:04, 15548.27it/s]
#1000061it [00:57, 17259.96it/s]
#1000061it [00:59, 16674.90it/s]
#100%|██████████| 4820416/4820416 [00:46<00:00, 104724.54it/s]
#1000061it [01:37, 10309.21it/s]
#Out[31]: 3.3701537969494333



