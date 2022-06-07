import pickle
import re
import pickle
import math
import random
import time
import datetime
from pathlib import Path
from os.path import join
import multiprocessing
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
import pandas as pd
import numpy as np
from protein_dataset import ProteinDataset
from jacob_temp_code.helper_functions import IUPAC_SEQ2IDX
import torch
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from Bio import SeqIO
from bioservices import UniProt
from Variations_of_VAE.models.vae import VAE_bayes_jaks
from Variations_of_VAE.helper_functions import *
from Variations_of_VAE.models import *
from Variations_of_VAE.data_handler import *


def parse_alignment_clusters(filename: str, base_path=None, order_alignment_file=None):
    """
    Parses mmseq cluster sequence fasta files.
    Each double Protein ID identifies cluster
    Impose order from original alignment file.
    """
    # load mmseq2 fasta output for clusters
    filepath = join(base_path, filename) if base_path else filename
    alignment_cluster_df = pd.read_csv(filepath, sep='\t', header=None)
    alignment_cluster_df.columns = ["Sequence"]

    cluster_list = []
    k = 0
    for i, entry in enumerate(alignment_cluster_df.Sequence[:-1]):
        cluster = f"cluster-{k}"
        cluster_list.append(cluster)
        # rename cluster identifiers from protein ID to cluster enumeration
        if entry == alignment_cluster_df.Sequence[i+1]:
            alignment_cluster_df.Sequence[i] = cluster # mark as cluster identifier
            k += 1
    cluster_list.append(cluster)
    cluster_mask = alignment_cluster_df.Sequence.str.startswith("cluster-")
    identifier_mask = alignment_cluster_df.Sequence.str.startswith(">")
    sequence_mask = ~(cluster_mask + identifier_mask)
    assert len(alignment_cluster_df[sequence_mask]) == len(alignment_cluster_df[identifier_mask]) == len(np.array(cluster_list)[sequence_mask])
    # print(f"Seq: {len(alignment_cluster_df[sequence_mask])} ; identifiers: {len(alignment_cluster_df[identifier_mask])} ; cluster_ids: {len(np.array(cluster_list)[sequence_mask])}" )
    formatted_cluster_df = pd.concat([pd.Series(np.array(cluster_list)[sequence_mask]),
                                      alignment_cluster_df[identifier_mask].Sequence.reset_index(drop=True),
                                      alignment_cluster_df[sequence_mask].Sequence.reset_index(drop=True)], axis=1)
    formatted_cluster_df.columns=["Cluster", "ID", "Sequence"]
    if order_alignment_file: # impose order by merging with orginal alignment order
      with open(order_alignment_file, "r") as infile:
        msa_alignment_ids = pd.DataFrame([entry.replace("\n", "") for entry in infile.readlines() if entry.startswith(">")], columns=["ID"])
      assert len(msa_alignment_ids) == len(formatted_cluster_df)
      formatted_cluster_df = msa_alignment_ids.merge(formatted_cluster_df, on="ID")
    else:
      raise Warning("Clustered sequences are different from MSA sequences of source DF!")
    return formatted_cluster_df


def sample_mmseq_clusters_validation_indices(cluster_df: pd.DataFrame, val_fraction=0.1):
  """
  Subsample at least one sample across all clusters uniformly until validation proportion of the MSE is reached.
  Note that the provided mmseq file contains MSA sequences only.
  """
  N_sequences = len(cluster_df)
  validation_indices = []
  while val_fraction > (len(validation_indices) / N_sequences):
    _indices = [cluster_df[cluster_df.Cluster==cluster].sample(n=1, random_state=42).index 
                   for cluster in cluster_df.Cluster.unique()]
    validation_indices += _indices
  validation_indices = np.array(validation_indices).flatten()
  return validation_indices

def get_datasets(data=None, train_ratio=0, device = None, 
                 SSVAE=False, SSCVAE=False, CVAE=False, regCVAE=False, 
                 train_with_assay_seq=False, only_assay_seqs=False, 
                 cluster_validation=False, cluster_file: str=None, cluster_order_file: str=None):
    seqs = len(data['seqs'])

    if cluster_validation and cluster_file:
        cluster_df = parse_alignment_clusters(cluster_file, order_alignment_file=cluster_order_file)
        val_seqs_indices = sample_mmseq_clusters_validation_indices(cluster_df)
        # drop val from training
        _msa_data = data[data.assay.isna()].copy()
        val_seqs = _msa_data.iloc[val_seqs_indices]
        train_seqs = _msa_data.drop(np.unique(val_seqs_indices), axis=0)
        print(f"Training data: {len(train_seqs)}")
        # DEBUG
        print(f"VAL: {len(val_seqs)}")
        print(f"TRAIN: {len(train_seqs)}")
        print(f"MSA: {train_seqs.head()}")

    all_data = ProteinDataset(data, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE, 
                              train_with_assay_seq=train_with_assay_seq, only_assay_seqs=only_assay_seqs)
    train_data = ProteinDataset(train_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE, 
                                train_with_assay_seq=train_with_assay_seq, 
                                only_assay_seqs=only_assay_seqs)
    val_data = ProteinDataset(val_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE,
                              train_with_assay_seq=train_with_assay_seq, only_assay_seqs=only_assay_seqs, training=False)
    return all_data, train_data, val_data


def get_protein_dataloader(dataset, batch_size = 128, shuffle = False, get_seqs = False, random_weighted_sampling = False):
    sampler = WeightedRandomSampler(weights = dataset.weights, num_samples = len(dataset.weights), replacement = True) if random_weighted_sampling else None
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle if not random_weighted_sampling else not random_weighted_sampling, collate_fn = seqs_collate, sampler = sampler)


def train_epoch(epoch, model, optimizer, scheduler, train_loader):
    """
        epoch: Index of the epoch to run
        model: The model to run data through. Forward should return a tuple of (loss, metrics_dict).
        optimizer: The optimizer to step with at every batch
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    train_loss = 0
    train_count = 0
    if scheduler is not None:
        learning_rates = []
    acc_metrics_dict = defaultdict(lambda: 0)
    for batch_idx, xb in enumerate(train_loader):
        batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)
        loss, batch_metrics_dict, px_z = train_batch(model, optimizer, xb, scheduler)
        # Model saves loss types in dict calculate accumulated metrics
        semisup_metrics =  ["seq2y_loss",
                            "z2y_loss",
                            "labelled seqs",
                            "unlabelled seqs",
                            "unlabelled_loss",
                            "labelled_loss"]
        for key, value in batch_metrics_dict.items():
            if key not in semisup_metrics:
                acc_metrics_dict[key] += value * batch_size
                acc_metrics_dict[key + "_count"] += batch_size
            if 'seq2y' in key or 'z2y' in key:
                if batch_metrics_dict['labelled seqs'] != None:
                    acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                    acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
                else:
                    acc_metrics_dict[key] += 0
                    acc_metrics_dict[key + "_count"] += 1
            if key == "unlabelled_loss":
                acc_metrics_dict[key] += value * batch_metrics_dict['unlabelled seqs'].size(0)
                acc_metrics_dict[key + "_count"] += batch_metrics_dict['unlabelled seqs'].size(0)
            if key == "labelled_loss" and batch_metrics_dict['labelled seqs'] != None:
                acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
            else:
                acc_metrics_dict[key] += 0
                acc_metrics_dict[key + "_count"] += 1
        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}
        train_loss += loss.item() * batch_size
        train_count += batch_size
        if scheduler is not None:
            learning_rates.append(scheduler.get_last_lr())
    average_loss = train_loss / train_count
    if scheduler is not None:
        metrics_dict['learning_rates'] = learning_rates
    return average_loss, metrics_dict, px_z


def train_batch(model, optimizer, xb, scheduler = None):
    model.train()
    # Reset gradient for next batch
    optimizer.zero_grad()
    # Push whole batch of data through model.forward() account for protein_data_loader pushes more than tensor through
    if isinstance(xb, Tensor):
        loss, batch_metrics_dict, px_z = model(xb)
    else:
        loss, batch_metrics_dict, px_z = model(*xb)
    # Calculate the gradient of the loss w.r.t. the graph leaves
    loss.backward()
    clip_grad_value = 200
    if clip_grad_value is not None:
        clip_grad_value_(model.parameters(), clip_grad_value)
    # Step in the direction of the gradient
    optimizer.step()

    # Schedule learning rate
    if scheduler is not None:
        scheduler.step()

    return loss, batch_metrics_dict, px_z

def train_vae(data_df, mmseq_alignment, hmmer_alignmnent_file):
    device = 'cuda'
    # determine VAE
    epochs = 150000
    latent_dim = 30
    name = 'ubqt'
    extra = 'test'
    df = data_df
    assay_df = df.dropna(subset=['assay']).reset_index(drop=True)
    random_weighted_sampling = True
    use_sparse_interactions = True
    use_bayesian = True
    use_param_loss = True
    batch_size = 100
    all_data, train_data, val_data = get_datasets(data=df,
                                                train_ratio=1,
                                                device = device,
                                                SSVAE=0,
                                                SSCVAE=0,
                                                CVAE=0,
                                                regCVAE=0,
                                                train_with_assay_seq=False, 
                                                only_assay_seqs=False,
                                                cluster_validation=True,
                                                cluster_file=MMSEQ_ALIGNMENT_FILE,
                                                cluster_order_file=ORIGINAL_ALIGNMENT_FILE)
    # prep downstream data
    def onehot_(arr):
        return F.one_hot(torch.stack([torch.tensor(seq, device='cpu').long() for seq in arr]), num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
    X_all_torch = torch.from_numpy(np.vstack(df['seqs'].values))
    X_labelled_torch = torch.from_numpy(np.vstack(assay_df['seqs'].values))
    # Construct dataloaders for batches
    train_loader = get_protein_dataloader(train_data, batch_size = batch_size, shuffle = False, random_weighted_sampling = random_weighted_sampling)
    val_loader = get_protein_dataloader(val_data, batch_size = batch_size)
    # log_interval
    log_interval = list(range(1, epochs, 5))

    # define input and output shape
    data_size = all_data[0][0].size(-1) * alphabet_size
    label_pred_layer_sizes = [0]
    z2yalpha = 0.01 * len(all_data)
    seq2yalpha = 0.01 * len(all_data)
    model = VAE_bayes(
        [data_size] + [1500, 1500, latent_dim, 100, 2000] + [data_size],
        alphabet_size,
        z_samples = 1,
        dropout = 0,
        use_bayesian = use_bayesian,
        use_param_loss = use_param_loss,
        use_sparse_interactions = use_sparse_interactions,
        rws=random_weighted_sampling,
        conditional_data_dim=0,
        SSVAE = 0,
        SSCVAE = 0,
        CVAE=0,
        VAE=1,
        regCVAE=0,
        multilabel=0,
        seq2yalpha=seq2yalpha,
        z2yalpha=z2yalpha,
        label_pred_layer_sizes = label_pred_layer_sizes,
        pred_from_latent=0,
        pred_from_seq=0,
        warm_up = 0,
        batchnorm=0,
        device = device
    )
    optimizer = optim.Adam(model.parameters())
    date = 'D'+str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)
    time = 'T'+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    date_time = date+time
    results_dict = defaultdict(list)
    VALIDATION_EPSILON = 10e-3 # convergence rate of validation
    validation_errors = [100000000] # initial error very high, mitigates early indexing issues
    overfitting_patience = 5 # validation intervals
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            model = model.cuda().float()
        start_time = datetime.datetime.now()
        train_loss, train_metrics, px_z = train_epoch(epoch = epoch, model = model, optimizer = optimizer, scheduler=None, train_loader = train_loader)
        val_str = ""
        results_dict['epochs'].append(epoch)
        results_dict['nll_loss_train'].append(train_metrics["nll_loss"])
        results_dict['kld_loss_train'].append(train_metrics["kld_loss"])
        results_dict['param_kld_train'].append(train_metrics["param_kld"])
        results_dict['total_train_loss'].append(train_loss)
        # print status
        print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Recon loss: {train_metrics['nll_loss']:.5f} KLdiv loss: {train_metrics['kld_loss']:.5f} Param loss: {train_metrics['param_kld']:.5f} {val_str}Time: {datetime.datetime.now() - start_time}", end="\n\n")
        if epoch in log_interval:
            model.eval()
            val_loss, _, _ = train_epoch(epoch = epoch, model = model, 
                                        optimizer = optimizer, scheduler=None, train_loader = val_loader)
            val_diff = np.abs(validation_errors[-1]-val_loss)
            print(f"Validation step: {epoch} - loss: {val_loss}, abs.diff={val_diff}")
            if val_diff <= VALIDATION_EPSILON or overfitting_patience == 0:
                print(f"ENDING TRAINING!")
                break
            if val_loss > validation_errors[-1]:
                print("Overfitted step... ")
                overfitting_patience -= 1
            else:
                overfitting_patience = 5
            validation_errors.append(val_loss)
            model.train()
    print('total epoch time', datetime.datetime.now()-start_time)
    with torch.no_grad():
        print('Saving...')
        pickle.dump( results_dict, open('VAE_CLUSTER_VAL'+extra+date_time+'_'+name+'_'+str(latent_dim)+'dim_final_results_dict.pkl', "wb" ) )
    # compute embedding from trained model on assay sequences:
    model.eval()
    mu_all_tmp = []
    for i, batch in enumerate(np.array_split(X_all_torch, math.ceil(len(X_all_torch)/1000))):
        mu_all, _ = model.cpu().encoder(batch, None)
        mu_all = mu_all.detach().numpy()
        mu_all_tmp.append(mu_all)
    mu_all = np.vstack(mu_all_tmp)
    with open(f'{name}_VAE_reps_CLUSTER_VAL.pkl', "wb") as outfile:
        pickle.dump(mu_all, outfile)



if __name__ == "__main__":
    DATASET_FILE = "./mth3_data_df.pkl"
    MMSEQ_ALIGNMENT_FILE = "./MTH3_mmseqs2_si02_ac08_seqs.fasta"
    ORIGINAL_ALIGNMENT_FILE = "./MTH3_HAEAESTABILIZED_1_b0.5.a2m"

    data_df = pickle.load(open(DATASET_FILE, "rb"))
    train_vae(data_df, MMSEQ_ALIGNMENT_FILE, ORIGINAL_ALIGNMENT_FILE)