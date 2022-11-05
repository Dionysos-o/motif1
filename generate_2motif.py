'''
Author: Chengbo Fu
Email: chengbo.fu@aalto.fi
'''
from ast import arg
from unicodedata import name
import numpy as np


def str2arr(dna_str: str) -> np.array:
    """
    convert dna sequences to numpy array
    :param dna_str: DNA string
    return: seq_arr
    """
    dna_str = dna_str.upper()
    mydict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    return np.array([mydict[x] for x in dna_str])


def hstack_random(seq, n, m):

    """
    Add random sites in both sides of the given seq
        
    """
    left = np.random.randint(4, size = (n,m))
    right= np.random.randint(4, size = (n,m))
    seq = np.tile(seq,(n, 1))
    seq_sum = np.hstack((left, seq))
    seq_sum = np.hstack((seq_sum, right))

    return seq_sum

    
def generate_kmer(seq, k):
    """

    convert sequence to k-mers

    :seq: dna sequence
    :k: length of kmer want to return
    """
    kmer = []

    for i, seq_i in enumerate(seq):
        for j in range(len(seq_i) - k + 1):
            kmer.append(seq_i[j: j+k].tolist())
    kmer = np.array(kmer)
    return kmer


def motif2_main(seq_c1, seq_c2, n, interval, kl):

    """

    generate a series of random DNA sequences include two given motifs 
    
    Arg: 
        seq_c1, seq_c2: two consensus sequnece
        n: the repeate times
        interval: the gap between motifs
        return: equal-length random sequences   
    
    """
    

    seq1_random = hstack_random(seq_c1, n, interval)
    seq2_random = hstack_random(seq_c2, n, interval)

    seq_random  = np.hstack((seq1_random, seq2_random))

    data = generate_kmer(seq_random, kl)
    
    return data


if __name__ == '__main__':

    seq1 = str2arr('AAGCGT')
    seq2 = str2arr('ATCTT')

    seq_random = motif2_main(seq1, seq2, 5, 3)
    print(seq_random.shape)


    kmer = generate_kmer(seq_random, 8)

    print(kmer.shape)



    
    


    