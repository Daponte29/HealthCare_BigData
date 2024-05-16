import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
import torch.nn as nn
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	df = pd.read_csv(path)
	# 提取特征和目标变量
	X = df.iloc[:, :-1].values # 假设最后一列是y
	y = df.iloc[:, -1].values - 1 # 将y的范围从1-5改为0-
	target = torch.tensor(y, dtype=torch.long)
	if model_type == 'MLP':
		data = torch.tensor(X, dtype=torch.float32)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		data = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset








def calculate_num_features(seqs):
    """
    :param seqs: List of List of List, representing visit sequences
    :return: the calculated number of features
    """
    # Flatten the list of lists
    flat_sequences = [code for visit in seqs for codes in visit for code in codes]

    # Get unique ICD9_CODES using set
    unique_icd9_codes = list(set(flat_sequences))
    return len(unique_icd9_codes)



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
        self.num_features = num_features
        self.seqs = []

        # Iterate over each patient's visit sequences
        for visit_sequence in seqs:
            seq_matrix = self.convert_to_matrix(visit_sequence, num_features)
            self.seqs.append(seq_matrix)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]

    def convert_to_matrix(self, visit_sequence, num_features):
        """
        Convert a visit sequence to a binary matrix representation.

        Args:
            visit_sequence (list): List of feature codes for a single patient's visit sequence.
            num_features (int): Number of total features available.

        Returns:
            numpy.ndarray: Binary matrix representation of the visit sequence.
        """
        # Initialize an empty matrix with zeros
        seq_matrix = np.zeros((len(visit_sequence), num_features), dtype=np.float32)

        # Iterate over each visit in the sequence
        for i, visit in enumerate(visit_sequence):
            # Iterate over each feature code in the visit
            for code in visit:
                if isinstance(code, (int, float)) and not np.isnan(code):
                    # Set the corresponding feature index to 1
                    seq_matrix[i, int(code)] = 1

        return seq_matrix



###NOTE:  
#ICD9 code starts with E then then first 4 characters and if not with E then first 3 AVER 900 or so 920 or 970.
#sparse matrix so each patient around 900 patients
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
    
    # Separate sequences, lengths, and labels
    seqs, labels = zip(*batch)
    
    # Sort the sequences and labels by the length of sequences in descending order
    seqs, labels = zip(*sorted(zip(seqs, labels), key=lambda x: len(x[0]), reverse=True))
    
    # Find the maximum sequence length in the batch
    max_length = max(len(seq) for seq in seqs)

    # Convert sequences to PyTorch tensors and pad them
    padded_seqs = [torch.nn.functional.pad(torch.tensor(seq), (0, 0, 0, max_length - len(seq))) for seq in seqs]

    # Convert padded sequences to a PyTorch tensor
    padded_seqs = torch.stack(padded_seqs)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.int64)  # Convert labels directly to a tensor with larger data type

    return (padded_seqs, torch.tensor([len(seq) for seq in seqs], dtype=torch.long)), labels_tensor

