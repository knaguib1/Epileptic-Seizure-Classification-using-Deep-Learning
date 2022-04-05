import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
    """
    :param path: a path to the seizure data CSV file
    :return dataset: a TensorDataset consists of a data Tensor and a target Tensor
    """
    # TODO: Read a csv file from path.
    # TODO: Please refer to the header of the file to locate X and y.
    # TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
    # TODO: Remove the header of CSV file of course.
    # TODO: Do Not change the order of rows.
    # TODO: You can use Pandas if you want to.
    
    # read csv file from path
    df = pd.read_csv(path)
    
    # extract labels
    # change data from 0 to 4
    y = df['y'].values - 1
    
    # extract features
    X = df.loc[:, 'X1':'X178'].values

    if model_type == 'MLP':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')),
                               torch.from_numpy(y.astype('long')))
        
    elif model_type == 'CNN':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')).unsqueeze(1),
                               torch.from_numpy(y.astype('long')))
        
    elif model_type == 'RNN':
        dataset = TensorDataset(torch.from_numpy(X.astype('float32')).unsqueeze(2), 
                            torch.from_numpy(y.astype('long')))
        
    else:
        raise AssertionError("Wrong Model Type!")

    return dataset


def calculate_num_features(seqs):
    """
    :param seqs:
    :return: the calculated number of features
    """
    # TODO: Calculate the number of features (diagnoses codes in the train set)
    nFeats = reduce(lambda x,y: x + y, seqs)
    nFeats = reduce(lambda x,y: x + y, nFeats)
    
    return max(nFeats) + 1


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels

        # TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
        # TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
        # TODO: You can use Sparse matrix type for memory efficiency if you want.
        
        feat = []
        
        for seq in seqs:
            
            m = np.zeros((len(seq), num_features))
            
            # i-th visit
            i = 0
            
            # iterate over each feature ID in sequence 
            for j in seq:
                
                # j-th column represent feature ID j
                m[i, j] = 1
                    
                i += 1
            # append results
            feat.append(m)
            
            self.seqs = feat  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """

    # TODO: Return the following two things
    # TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
    # TODO: 2. Tensor contains the label of each sequence
        
    tensor_seq = [(item[0].shape[0], i) for i, item in enumerate(batch)]
    tensor_seq.sort(key = lambda k: k[0], reverse=True)
    
    i = tensor_seq[0][0]
    j = batch[0][0].shape[1]
    
    seq = []
    nTensor = []
    labels = []
    
    tensor_keys = [t[1] for t in tensor_seq]
    
    for k in tensor_keys:
    
        data = batch[k]
        
        # lengths of tensor
        nRow = data[0].shape[0]
        nCol = data[0].shape[1]
        nTensor.append(nRow)
        
        # labels of tensor
        labels_tmp = data[1]
        labels.append(labels_tmp)
        
        # construct matrix        
        m = np.zeros((i, j))
        m[:nRow, :nCol] = data[0]
        
        seq.append(m)
    
    seq = np.array(seq)
    nTensor = np.array(nTensor)
    labels = np.array(labels)
        
    seqs_tensor = torch.FloatTensor(seq)
    lengths_tensor = torch.LongTensor(nTensor)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor
