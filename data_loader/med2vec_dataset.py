#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch
import torch.utils.data as data
import os
import pickle
import numpy as np

class Med2VecDataset(data.Dataset):

    def __init__(self, root, num_codes, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.num_codes = num_codes
        if download:
            raise ValueError('cannot download')

        self.train_data = pickle.load(open(root, 'rb'))
        self.test_data = []
        print("train and validation data from pickle is:",self.train_data[:50])

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        x, ivec, jvec, d = self.preprocess(self.train_data[index])
        return x, ivec, jvec, d

    def preprocess(self, seq):
        # creating the new sentences where ivec will be the target words and the jvec will be all the contex words around the target word'
        # in ivec each word will be duplicated b the amount of words in its relevant context word

        """ create one hot vector of idx in seq, with length self.num_codes

            Args:
                seq: list of indices where code should be 1

            Returns:
                x: one hot vector for each word in the batch
                ivec: vector for learning code representation
                jvec: vector for learning code representation
        """
        x = torch.zeros((self.num_codes, ), dtype=torch.long)
        ivec = []
        jvec = []
        d = []
        # when visit is [-1] it means its end of patient
        if seq == [-1]:
            return x, torch.FloatTensor(ivec), torch.FloatTensor(jvec), torch.FloatTensor(d)

        x[seq] = 1
        for i in seq:
            for j in seq:
                if i == j:
                    continue
                ivec.append(i)
                jvec.append(j)
        return x, torch.FloatTensor(ivec), torch.FloatTensor(jvec), torch.FloatTensor(d)

def collate_fn(data):
    """ Creates mini-batch from x, ivec, jvec tensors
    removes all the false words in the sentence - all the words that are used as a symbole of end of person [-1]

    We should build custom collate_fn, as the ivec, and jvec have varying lengths. These should be appended
    in row form

    Args:
        data: list of tuples contianing (x, ivec, jvec)

    Returns:
        x: one hot encoded vectors stacked vertically
        ivec: long vector
        jvec: long vector
    """

    x, ivec, jvec, d = zip(*data)
    x = torch.stack(x, dim=0)
    d = torch.stack(d, dim=0)
    mask = torch.sum(x, dim=1) > 0
    mask = mask[:, None]
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    return x, ivec, jvec, mask, d

def get_loader(root, num_codes, train=True, transform=None, target_transform=None, download=False, batch_size=8):
    """ returns torch.utils.data.DataLoader for Med2Vec dataset """
    shuffle = False # define shuffle variable
    num_workers = 4 # define num_workers variable
    med2vec = Med2VecDataset(root, num_codes, train, transform, target_transform, download)
    data_loader = torch.utils.data.DataLoader(dataset=med2vec, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
