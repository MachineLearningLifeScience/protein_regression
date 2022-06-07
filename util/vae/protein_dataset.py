import pandas as pd
import torch
from Bio import SeqIO
from bioservices import UniProt
from jacob_temp_code.helper_functions import IUPAC_AMINO_IDX_PAIRS, IUPAC_SEQ2IDX

class ProteinDataset(Dataset):
    def __init__(self, data, device = None, SSVAE=False, SSCVAE=False, CVAE=False, regCVAE=False,
                 train_with_assay_seq=False, only_assay_seqs=False, training=True):
        super().__init__()
        if len(data) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return

        if training:
            actual_labels = data[~data.assay.isna()]
            train_labelled_data = actual_labels
        if 'blast' in data.columns and not train_with_assay_seq:
            data = data.drop(data.index[data['blast'] != 1].tolist(), axis=0)
        if 'blast' not in data.columns and training:
            data = data.drop(actual_labels.index, axis=0)

        if training:
            if SSVAE or CVAE or SSCVAE or regCVAE or train_with_assay_seq:
                data = pd.concat((data, train_labelled_data), axis=0)

        if only_assay_seqs:
            data = train_labelled_data
        # START DEBUG
        print("SEQS")
        print(data['seqs'].shape)
        # END DEBUG
        self.encoded_seqs = torch.stack([torch.tensor(seq, device=device) for seq in data['seqs'].values]).long()
        num_sequences = self.encoded_seqs.size(0)
        self.labels = torch.Tensor()

        discretize = True if SSCVAE or CVAE else False

        if not SSVAE and not CVAE and not SSCVAE and not regCVAE:
            self.labels = None

        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
        weight_batch_size = 1000
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()
        print('Neff', self.neff)
    def write_to_file(self, filepath):
        for s, w in zip(self.seqs, self.weights):
            s.id = s.id + ':' + str(float(w))
        SeqIO.write(self.seqs, filepath, 'fasta')

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        if self.labels == None:
            labels = self.labels
        else:
            labels = self.labels[i]
        return self.encoded_seqs[i], self.weights[i], self.neff, labels