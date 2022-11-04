import random
from array import array
from ast import Str
from tkinter import N
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from generate_2motif import motif2_main, generate_kmer
from Levenshtein import distance as lev
import copy
def gen_seqs(c_seq: np.array, n_seq: int):
    """
    generate DNA given a consensus sequence
    :param c_seq: consensus seq, int array of [0, 1, 2, 3] <=> ACGT
    :param n_seq: number of seq to be generated
    :return: seq_mat, n_seq x k
    """
    c_seq = np.array(c_seq)
    k = len(c_seq)
    prob = [(k-1)/k, 1/(3*k), 1/(3*k), 1/(3*k)]  # one mutation per sequence on average, randomly advance by 0, 1, 2, 3
    # print(sum(prob))
    seq_mat = np.random.choice(n_seq, k, replace=True, p=prob) + c_seq
    seq_mat = seq_mat % 4
    return seq_mat

def gen_motifs(c_seq: np.ndarray, n_seq: int):
    """
    generate motif given a consensus sequence
    :param c_seq:
    :param n_seq:
    :return:
    """
    c_seq = np.array(c_seq)
    k = len(c_seq)
    position = np.random.choice(k, 300)
    motif = []
    for i in range(len(position)):
        tem_c_seq = copy.deepcopy(c_seq)
        tem_c_seq[position[i]] = (tem_c_seq[position[i]] + np.random.randint(1, 3)) % 4
        motif.append(tem_c_seq)
    n_con_seqs = np.tile(c_seq, (300, 1))

    return np.hsteck(np.array(motif),n_con_seqs)



def gen_seq_motif(motif_mat: np.array, n_seq: int):
    """
    generate DNA given a motif matrix
    :param motif_mat: motif matrix, k x 4
    :param n_seq: number of sequences
    :return: seq_mat, n_seq x k
    """
    assert motif_mat.shape[1] == 4
    k = motif_mat.shape[0]
    motif_mat = motif_mat / np.sum(motif_mat, axis=1, keepdims=True)
    seq_mat = np.zeros((n_seq, k), dtype=int)
    for i, prob in enumerate(motif_mat):
        seq_mat[:, i] = np.random.choice(4, n_seq, replace=True, p=prob)
    return seq_mat


def trans_array_str(a: int, b: np.array) -> str:
    """
    
    """
    k = a + 1
    m = np.ones(k) + b
    st = len(m)*'a'
    return st


def mat2str(seq_mat: np.array):
    """
    Convert seqmat into a string list
    :param seq_mat: numpy int array
    :return: list
    """
    if seq_mat.ndim == 1:
        return ''.join(seq_mat)
    myfun = np.vectorize(lambda x: 'ACGT'[x])
    return [''.join(row) for row in myfun(seq_mat)]


def str2arr(dna_str: str) -> np.array:
    """
    convert dna sequences to numpy array
    :param dna_str: DNA string
    :return: seq_arr
    """
    dna_str = dna_str.upper()
    mydict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mydict[x] for x in dna_str])


def arr2str(dna_arr: array) -> str:

    mydict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    str1 = ''
    for i in dna_arr:
        str1 = str1 + mydict[i]
    return str1
    

def cal_prob(seq:np.array,motif_mat:np.array):
    prob = np.zeros(len(seq))
    for j, i in enumerate(seq):
        prob[j] = motif_mat[j][i] /np.sum(motif_mat[j])

    pr = -np.log(np.prod(prob))

    return pr


def hamming_distance(x, y):
    """
    calculate the hamming distance between two kmers
    """
    count = 0

    for i, xi in enumerate(x):
        if xi != y[i]:
            count = count + 1

    return count


def gen_label(data, con_seq):

    """
    calculate the distance from consensus sequence,

    """
    label = np.zeros(len(data))
    for i in range(len(data)):
        label[i] = hamming_distance(data[i], con_seq)
    return label

def gen_label_2motif(data, con_seq1, con_seq2):

    label = np.zeros(len(data))
    for i in range(len(data)):
        dis1 = hamming_distance(data[i], con_seq1)
        dis2 = hamming_distance(data[i], con_seq2)
        if dis1 <= 1:
            label[i] = 1
        elif dis2 <= 1:
            label[i] = 2
        else:
            label[i] = 3
    return label


def gen1(kl, n_seq):

    """
    one motif with random sequence

    """
    c_seq1 = str2arr("AACGTATT")
    data1_part1 = gen_seqs(c_seq1, n_seq)
    data1_random = np.random.randint(4, size=(1000, kl))
    data1 = np.vstack([data1_part1, data1_random])
    label1 = np.zeros(len(data1))
    label1[:1000] = 1
    label = gen_label(data1, c_seq1)
    return data1, label


def gen2():
    """
    generate two motif data
    """
    n_seq = 1000
    c_seq1 = str2arr("AACGTA")
    c_seq2 = str2arr("AGAACC")
    data1_part1 = gen_seqs(c_seq1, n_seq)
    data1_part2 = gen_seqs(c_seq2, n_seq)

    data = np.vstack((data1_part1, data1_part2))

    return data


def gen3():
    """
    60 motif sequence (hamdis<=1), 40 random sequence(hamdis>=2), 
    For any x in random sequence, |x-c| >= 2
    """
    n_seq = 600
    c_seq1 = str2arr("AACTGTCCA")
    data1_part = gen_seqs(c_seq1, n_seq)
    data1_random = np.random.randint(4, size=(1000, len(c_seq1)))
    data1_random_pick = []
    for x in data1_random:
        if hamming_distance(x, c_seq1) >=2:
            data1_random_pick.append(x)
        if len(data1_random_pick) == n_seq*2/3:
            break
    data1_random_pick = np.array(data1_random_pick)
    data = np.vstack((data1_part,data1_random_pick))
    label = np.zeros(len(data))
    for i in range(len(data)):
        distance = hamming_distance(data[i], c_seq1)
        if distance == 1:
            label[i] = 1
        elif distance >=2:
            label[i] = 2
    return data, label


def gen4():
    """
    generate sequence data include two motif

    """
    c_seq1 = str2arr("AACGTA")
    c_seq2 = str2arr("AGAACC")
    n_seq = 100
    interval = 5
    kl = 4
    data = motif2_main(c_seq1, c_seq2, n_seq, interval, kl)
    label = gen_label_2motif(data, c_seq1, c_seq2)
    return data, label

def gen5_com(data):
    """
    For 9mer, mutation at first and the end to produce 11mer
    """

    mutation = np.random.randint(4, size=(len(data), 2))
    first = mutation[:, 0]
    end = mutation[:, 1]
    # insert at both first and last position
    data1_part1 = np.insert(data, 0, values=first, axis=1)
    data1_part2 = np.insert(data1_part1, 10, values=end, axis=1)
    # insert at first position
    data2 = np.insert(data, 0, values=first, axis=1)
    data2 = np.insert(data2, 0, values=end, axis=1)
    # insert at last position
    data3 = np.insert(data, 9, values=first, axis=1)
    data3 = np.insert(data3, 10, values=end, axis=1)


    data = np.vstack((data1_part2,data2,data3))

    data_7mer = mat2str(generate_kmer(data, 7))
    data_9mer = mat2str(generate_kmer(data, 9))
    data_11mer = mat2str(generate_kmer(data, 11))
    data_kmer = np.concatenate((data_7mer, data_9mer, data_11mer))

    return data_kmer


def label_sub(data_part, con_seq, sym):
    label_part = []
    for i in range(len(data_part)):
        dis = hamming_distance(data_part[i], con_seq)
        if dis == 0:
            label_part.append(sym)
        else:
            label_part.append(sym + 1)
    return np.array(label_part)

def gen5():
    """
    generate three motif data in 9mer
    """
    n_seq = 600
    c_seq1 = str2arr("AACTGTCCA")
    c_seq2 = str2arr("ATCCGCTAC")
    c_seq3 = str2arr("ACTACTCCC")

    data1_part = gen_motifs(c_seq1, n_seq)
    data2_part = gen_motifs(c_seq2, n_seq)
    data3_part = gen_motifs(c_seq3, n_seq)
    data_random = np.random.randint(4, size=(1000, len(c_seq1)))
    data_random_pick = []

    for x in data_random:
        dis1 = hamming_distance(x, c_seq1)
        dis2 = hamming_distance(x, c_seq2)
        dis3 = hamming_distance(x, c_seq3)
        if min(dis1, dis2, dis3) >= 2:
            data_random_pick.append(x)
        if len(data_random_pick) == int(n_seq * 2 / 3):
            break
    data_random_pick = np.array(data_random_pick)

    """generate kmers data and it's label"""



    """generate motif data and it's label,neighbor label"""

    data = np.vstack((data1_part, data2_part, data3_part, data_random_pick))
    #data_kmer = gen5_com(data)


    label_part1 = label_sub(data1_part, c_seq1, 0)
    label_part2 = label_sub(data2_part, c_seq2, 2)
    label_part3 = label_sub(data3_part, c_seq3, 4)
    label_part4 = np.zeros(len(data_random_pick)) + 6
    label_1 = np.concatenate((label_part1, label_part2, label_part3, label_part4))


    label1 = np.zeros(n_seq)
    label2 = np.zeros(n_seq) + 1
    label3 = np.zeros(n_seq) + 2
    label4 = np.zeros(len(data_random_pick)) + 3
    label_2 = np.concatenate((label1, label2, label3, label4))

    return data, label_1, label_2

def levenshtein(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t:
        return 0
    elif len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
    return v1[len(t)]

def cal_levenshtein(x):
    dist = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            dist[i][j] = levenshtein(x[i], x[j])
    return dist


def simulation_run_1():
    """
    simulate one dataset with two similar consensus sequence "AACGTA", "GACGAA"
    and one dataset that combines these two using motif model
    :return: data1 and data2
    """
  

    #data1 = generate_kmer(motif2_main(c_seq1, c_seq2, n_seq, interval), kl)

    #data_kmer, label_kmer =gen5()

    data1, label1, label2 = gen5()
    data1 = torch.from_numpy(data1)
    label1 = torch.from_numpy(label1)
    label2 = torch.from_numpy(label2)
    data1 = F.one_hot(data1, num_classes=4)
    data1 = np.reshape(data1, (len(data1), data1.shape[2]*data1.shape[1]))

    return data1, label1, label2


if __name__ == '__main__':
    kmer = gen5()
    np.random.shuffle(kmer)

    print(cal_levenshtein(kmer))

    #data_, label_ = simulation_run_1(3, 4, 5)
    #print(data_, label_)








    '''
    data1, pro, n_seq = simulation_run_1(6,300,5)


    pro1 = pd.DataFrame(pro)
    pro1.plot.box()
    plt.show()


    index = np.argwhere(pro < 4.25)
    new_pro = np.delete(pro,index)
    
            
    pro1 = pd.DataFrame(new_pro)
    pro1.plot.box()
    plt.show()
    

    plt.hist(pro,bins=40, density = True, stacked=True, facecolor = 'blue',edgecolor = 'black',alpha = 0.7)
    plt.show()

    x = list(range(len(pro)))

    plt.plot(x,pro,'r--','type1')
    plt.show()


    x,fre =np.unique(data1, axis = 0, return_counts=True)
    print(len(x)) 
    plt.hist(pro,bins=40, density = True, stacked=True, facecolor = 'blue',edgecolor = 'black',alpha = 0.7)
    plt.show()
    '''