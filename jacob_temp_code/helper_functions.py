# %%
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
import pickle
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from models import *



def sampling(z_samples, state_dict, assay_df, assay, layer_sizes, temperature=1, alphabet_size=24, labels = [None], rws=True, multilabel = 0, conditional_data_dim=0,
                        seq2yalpha = 0.01 * 5200, z2yalpha = 0.01 * 5200, pred_from_latent = False,
                        pred_from_seq = False, batchnorm = False, ensemble_count=500, device='cpu',
                        use_bayesian = True, use_sparse_interactions = True, use_param_loss = True):

    seqs = torch.stack([torch.tensor(seq, device=device) for seq in assay_df['seqs']])
    model = VAE_bayes(
        layer_sizes,
        alphabet_size,
        z_samples = z_samples,
        use_bayesian = use_bayesian,
        use_param_loss = use_param_loss,
        use_sparse_interactions = use_sparse_interactions,
        rws=rws,
        conditional_data_dim=conditional_data_dim,
        SSVAE = 0,
        SSCVAE = 0,
        CVAE= 0,
        VAE= 1,
        regCVAE=0,
        multilabel=multilabel,
        seq2yalpha=seq2yalpha,
        z2yalpha=z2yalpha,
        label_pred_layer_sizes = [],
        pred_from_latent=pred_from_latent,
        pred_from_seq=pred_from_seq,
        batchnorm=batchnorm,
        device = device,
        warm_up=0
    )

    model.load_state_dict(state_dict['model_state_dict'])

    with torch.no_grad():
        x = seqs #repeat_interleave

        y = np.tile(assay_df[assay].values, z_samples).astype(float) #repeat

        mu, logvar = model.encoder(seqs, labels)
        qz_x = Normal(mu, logvar.mul(0.5).exp())
        prior = Normal(torch.zeros_like(qz_x.mean), torch.ones_like(qz_x.variance))

        std = logvar.mul(0.5).exp() * temperature
        z = Normal(mu, std).rsample((z_samples,))
        z_prior = prior.rsample((z_samples,))

    return z.flatten(0,1).cpu().numpy(), y



def onehot_(arr):
    return F.one_hot(torch.stack([torch.tensor(seq, device='cpu').long() for seq in arr]), num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1).cpu().numpy()


IUPAC_IDX_AMINO_PAIRS = list(enumerate([
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<mask>",
    'B'
]))
IUPAC_AMINO_IDX_PAIRS = [(a, i) for (i, a) in IUPAC_IDX_AMINO_PAIRS]
alphabet_size = len(IUPAC_AMINO_IDX_PAIRS)
IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS)

def calc_seq_weights(encoded_seqs, device):
    weights = []
    encoded_seqs = torch.stack([torch.tensor(seq, device=device) for seq in encoded_seqs])
    flat_one_hot = F.one_hot(encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
    weight_batch_size = 1000
    for i in range(encoded_seqs.size(0) // weight_batch_size + 1):
        x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
        similarities = torch.mm(x, flat_one_hot.T)
        lengths = (encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
        w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
        weights.append(w)
    weights = torch.cat(weights)
    neff = weights.sum()

    return weights, neff 


def seqs_collate(tensors):
    encoded_seq, weights, neffs = zip(*tensors)
    return torch.stack(encoded_seq), neffs[0], torch.stack(weights)


def get_protein_dataloader(dataset, batch_size = 100, shuffle = False, weighted_sampling = False):
    sampler = WeightedRandomSampler(weights = dataset.weights, num_samples = len(dataset.weights), replacement = True) if weighted_sampling else None
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle if not weighted_sampling else not weighted_sampling, collate_fn = seqs_collate, sampler = sampler)


def regularCV_pred(X, y, splits, modeltype=RandomForestRegressor, seed = 42):
        
    mse = []
    mse_base = []
    kf = KFold(n_splits=splits, random_state=seed, shuffle=True)
    counter = 0
    
    for train_idx, test_idx in kf.split(X):
        counter += 1
        print('Fold:', counter,'/',splits)
        X_train, y_train = X[train_idx], y[train_idx].values
        X_test, y_test = X[test_idx], y[test_idx].values

        import multiprocessing
        num_cpu = multiprocessing.cpu_count()
        model = modeltype(random_state=seed, n_jobs=num_cpu)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        mse_test = mean_squared_error(pred, y_test)
        mse.append(mse_test)
        
        base = mean_squared_error(np.tile(np.mean(y_train),len(y_test)), y_test)
        mse_base.append(base)

    return mse, np.mean(mse_base)



def positional_splitter(assay_df, query_seqs, val=True, aln_path=None, offset = 4, pos_per_fold = 100, 
                        threshold = 2, split_by_DI = False):
    # offset is the positions that will be dropped between train and test positions 
    # to not allow info leakage just because positions are neighbours
    # Split_by_DI implements that positions are also split by direct info based on a "threshold"
    # needs an aln_path to work (fasta file)
    mut_pos = []
    for seq in assay_df['seqs']:
        mut_pos.append(np.argmax(query_seqs!=seq))
    index = list(range(len(mut_pos)))
    unique_mut = np.unique(mut_pos)
    
    #if split_by_DI:
     #   with open(aln_path, "r") as infile:
      #      aln = Alignment.from_file(infile, format="fasta")
#
 #       df_couplings = MeanFieldDCA(aln).fit()._calculate_ecs()

    train_indices = []
    test_indices = []
    val_indices = []

    counter = 0
    print(len(np.unique(mut_pos))//(pos_per_fold)+1)
    for i in range(len(np.unique(mut_pos))//(pos_per_fold)+1):  
        
        test_mut = list(unique_mut[counter:counter+pos_per_fold])
        if len(test_mut)==0:
            continue
        
        train_mut = list(unique_mut[:max(counter-offset, 0)]) +\
                    list(unique_mut[counter+pos_per_fold+offset:])
        
        if offset > 0:
            buffer_mut = list(unique_mut[max(counter-offset, 0):counter]) +\
                     list(unique_mut[counter+pos_per_fold:counter+pos_per_fold+offset])
        else:
            buffer_mut = []

        #if split_by_DI:
        #    interacting_pos = []
        #    for pos in np.unique(test_mut):
        #        if pos in df_couplings[(np.abs(df_couplings['cn'])>threshold)]['i'].values:
        #            interacting_pos.append(pos)
        #        if pos in df_couplings[(np.abs(df_couplings['cn'])>threshold)]['j'].values:
        #            interacting_pos.append(pos)
#
        #    interacting_pos = np.unique(interacting_pos)
        #    interacting_idx = np.array([np.where(mut_pos == pos) for pos in interacting_pos]).flatten()
#
        #    test_mut = [x for x in test_mut if x not in interacting_pos]
        #    train_mut = train_mut + [x for x in interacting_pos if x not in test_mut]
        
        if val:
            print(list(np.hstack([unique_mut[:int(1/6*pos_per_fold)], unique_mut[-int(1/6*pos_per_fold):]])))
            val_mut = [unique_mut[-int(1/3*pos_per_fold):],
                      np.hstack([unique_mut[:int(1/6*pos_per_fold)], unique_mut[-int(1/6*pos_per_fold):]]),
                      unique_mut[:int(1/3*pos_per_fold)]]
            train_mut = [mut for mut in train_mut if mut not in val_mut[i]]
        else:
            val_mut = [[] for i in range(len(np.unique(mut_pos)))]

        test_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in test_mut])
        train_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in train_mut])

        if offset>0:
            buffer_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in buffer_mut])
        else:
            buffer_idx = []
        
        if val:
            val_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in val_mut[i]])
        else:
            val_idx = [[] for i in range(len(np.unique(mut_pos)))]
        
        gaps = 8 if counter-offset >= 0 and len(unique_mut)-(counter+pos_per_fold) >= 0 else 4
        verify_num_mut = len(test_mut) + len(train_mut) + len(buffer_mut) + len(val_mut[i])
        verify_num_idx = (len(test_idx) + len(train_idx) + len(buffer_idx)) + len(val_idx)
        assert len(list(set(test_mut).intersection(set(train_mut))))==0, "test and train idx overlap"
        assert len(list(set(train_idx).intersection(set(test_idx))))==0, "test and train idx overlap"
        assert len(unique_mut) == verify_num_mut, 'Something wrong with number of positions/mutations. Number of idx: '+\
                                                    str(verify_num_idx) + 'Number of mut:' + str(verify_num_mut)
        
        train_indices.append(train_idx)
        val_indices.append(val_idx)
        test_indices.append(test_idx)

        counter += pos_per_fold
        
    return train_indices, val_indices, test_indices

def pred_func(X, y, train_indices, test_indices, modeltype=RandomForestRegressor, seed = 42):
    mse = []
    mse_base = []
    for i in range(len(train_indices)):
        print('Fold:', i+1,'/',len(train_indices))
        train_idx = train_indices[i]
        test_idx = test_indices[i]

        X_train, y_train = np.vstack(X[train_idx]), y[train_idx].values
        X_test, y_test = np.vstack(X[test_idx]), y[test_idx].values

        import multiprocessing
        num_cpu = multiprocessing.cpu_count()
        model = modeltype(random_state=seed, n_jobs=num_cpu)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse_test = mean_squared_error(pred, y_test)
        mse.append(mse_test)
        base = mean_squared_error(np.tile(np.mean(y_train), len(y_test)), y_test)
        mse_base.append(base)

    mse_mean_base = np.mean(mse_base)

    return mse, np.mean(mse_mean_base)

def pos_per_fold_assigner(name):
    if name=='ubqt':
        pos_per_fold = 25
    if name=='blat':
        pos_per_fold = 85
    if name=='brca':
        pos_per_fold = 63
    if name=='timb':
        pos_per_fold = 28
    if name=='calm':
        pos_per_fold = 47
    if name=='mth3':
        pos_per_fold = 107
    return pos_per_fold


def max_len_finder(name):
    if name.lower() == 'blat':
        threshold = 550
    if name.lower() == 'brca':
        threshold = 2000
    if name.lower() == 'calm':
        threshold = 750
    if name.lower() == 'timb':
        threshold = 550
    if name.lower() == 'mth3':
        threshold = 1000
    return threshold

def prep_data(D, max_len):
    X = [torch.from_numpy(x) for x in D]
    X = [ConstantPad1d((0,max_len-x.shape[0]),0)(x.T) for x in X]
    X = torch.stack(X)
    return X


class CustomDataset(Dataset):
    def __init__(self, X, encoded_seqs, device):
        self.X = X.float().cpu()
        self.weights, self.neff = calc_seq_weights(encoded_seqs, device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.weights[idx], self.neff

if __name__ == "__main__":
    name = 'blat'    
    df = pickle.load( open( 'data/'+name.lower()+'_seq_reps_n_phyla.pkl', "rb" ) )
    query_seqs = df['seqs'][0]
    assay_df = df.dropna(subset=['assay']).reset_index(drop=True)
    y = assay_df['assay']
    cv_folds = 10
    offset = 4
    split_by_DI = False # if True set a threshold
    pos_per_fold = pos_per_fold_assigner(name)
    train_indices, val_indices, test_indices = positional_splitter(assay_df, query_seqs, val=True, offset = offset, pos_per_fold = pos_per_fold, 
                                                split_by_DI = split_by_DI)
