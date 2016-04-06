import re
import numpy as np

lfs = []

##### UTILS #####
def is_gene_parent(c, key):
    return key in [c.lemmas[c.dep_parents[i] - 1] for i in c.e1_idxs]
def pre_window(c, key, match='e1_idxs', n=3):
    s = list(c.__dict__[match])
    b = np.min(s)
    s.extend([b - i for i in range(1, min(b,n+1))])
    return key in [c.lemmas[i] for i in s]
def post_window(c, key, match='e1_idxs', n=3):
    s = list(c.__dict__[match])
    b = len(c.lemmas) - np.max(s)
    s.extend([np.max(s) + i for i in range(1, min(b,n+1))])
    return key in [c.lemmas[i] for i in s]
def span_len(c):
    return np.min(c.e2_idxs) - np.max(c.e1_idxs)
def stopper(c, stop, match='e1_idxs'):
    return stop in [c.lemmas[i] for i in c.__dict__[match]]

##### POSITIVE RULES #####

prog_1 = re.compile(r"\{\{G\}\}(.*)(cause|responsible for)(.*)\{\{P\}\}")
prog_2 = re.compile(r"\{\{G\}\}(.*)in patients with(.*)\{\{P\}\}")
prog_3 = re.compile(r"\{\{P\}\}(.*)(caused by|due to|result of|attributable to)(.*)\{\{G\}\}")
prog_4 = re.compile(r"\{\{P\}\}results from(.*)\{\{G\}\}")

def lf_mutation(c):
    return 1 if is_gene_parent(c, 'mutation') else 0
def lf_deletion(c):
    return 1 if pre_window(c, 'deletion') else 0
def lf_loss(c):
    return 1 if pre_window(c, 'loss') else 0
def lf_short(c):
    return 1 if np.abs(span_len(c)) < 5 else 0
lfs.extend([lf_mutation, lf_deletion, lf_loss, lf_short]) 

def lf_p_1(c):
    return 1 if prog_1.search(c.tagged_sent) else 0
def lf_p_2(c):
    return 1 if prog_2.search(c.tagged_sent) else 0
def lf_p_3(c):
    return 1 if prog_3.search(c.tagged_sent) else 0
def lf_p_4(c):
    return 1 if prog_4.search(c.tagged_sent) else 0 
lfs.extend([lf_p_1, lf_p_2, lf_p_3, lf_p_4])  

##### NEGATIVE RULES #####

prog_5 = re.compile(r"\{\{G\}\}(.*)(whereas|however|not)(.*)\{\{P\}\}")
prog_6 = re.compile(r"\{\{P\}\}(.*)(whereas|however|not)(.*)\{\{G\}\}")
prog_7 = re.compile(r"\{\{G\}\}(.*)_(.*)_(.*)\{\{P\}\}")
prog_8 = re.compile(r"\{\{P\}\}(.*)_(.*)_(.*)\{\{G\}\}")

def lf_long(c):
    return -1 if np.abs(span_len(c)) > 25 else 0
def lf_protein(c):
    return -1 if post_window(c, 'protein', n=2) else 0
def lf_express(c):
    return -1 if (post_window(c, 'express', n=2) or pre_window(c, 'express', n=2)) else 0
def lf_all(c):
    return -1 if stopper(c, 'all', 'e2_idxs') else 0
def lf_aut(c):
    return -1 if (' '.join([c.lemmas[i] for i in c.e2_idxs]).lower() == 'autosomal dominant') else 0
def lf_x(c):
    return -1 if (' '.join([c.lemmas[i] for i in c.e2_idxs]).lower() == 'x-linked') else 0
lfs.extend([lf_long, lf_protein, lf_express, lf_all, lf_aut, lf_x])    
    
def lf_p_5(c):
    return 1 if prog_5.search(c.tagged_sent) else 0
def lf_p_6(c):
    return 1 if prog_6.search(c.tagged_sent) else 0
def lf_p_7(c):
    return 1 if prog_7.search(c.tagged_sent) else 0
def lf_p_8(c):
    return 1 if prog_8.search(c.tagged_sent) else 0 
lfs.extend([lf_p_5, lf_p_6, lf_p_7, lf_p_8])  
