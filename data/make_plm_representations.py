"""
This code wraps up all code that can be found in the Notebooks:
    esm_representation.ipynb
    prot5_representation.ipynb
To be used to derive embeddings on cluster compute resources.
"""
from itertools import product
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from os.path import join
from tqdm import tqdm
import pickle
from typing import Tuple, List
import torch
from data.load_dataset import get_wildtype_and_offset
from data.get_alphabet import get_alphabet
from data.load_dataset import load_one_hot


AVAILABLE_MODELS = {
    "esm1b": "esm1b_t33_650M_UR50S",
    "esm1v": "esm1v_t33_650M_UR90S_1",
    "esm2": "esm2_t36_3B_UR50D",
    }
ALL_DATASETS = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"]


def compute_pll(sequences, model, batch_tokens, alphabet, device):
    log_ps = []
    for s in range(len(sequences)):
        seq_ll = []
        for i in range(1, len(sequences[s])): # NOTE: skipping cls start token
            _batch_tokens_masked = batch_tokens.clone()
            _batch_tokens_masked[s, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(model(_batch_tokens_masked.to(device))["logits"], dim=-1)
            # NOTE: sequences is a 1D array of strings, double index required
            #print(f"ll s={s} at i={i} = {token_probs[s, i, alphabet.get_idx(sequences[s][i])].item()}")
            seq_ll.append(token_probs[s, i, alphabet.get_idx(sequences[s][i])].item())
        log_ps.append(sum(seq_ll))
    return log_ps


def compute_esm_plls(model, converted_seqs, batch_converter, alphabet, dev, stepsize: int=10) -> np.ndarray:
    print("Compute PLL ...")
    log_ps = []
    with torch.no_grad():
        for i in tqdm(range(0, len(converted_seqs), stepsize)):
            batched_sequences = converted_seqs[i:i+stepsize]
            data = [(f"protein{j}", seq) for j, seq in enumerate(batched_sequences)]
            _, _, batch_tokens = batch_converter(data)
            log_probs = compute_pll(batched_sequences, model, batch_tokens, alphabet, device=dev)
        log_ps.append(log_probs)
    ll_array = np.vstack(np.array(log_ps))
    return ll_array


def compute_esm_representation(model, converted_seqs, batch_converter, alphabet, dev, stepsize: int=10, rep_lyr: List[int]=[33]) -> np.ndarray:
    rep = []
    # Extract per-residue representations and limit memory requirements
    with torch.no_grad():
        for i in tqdm(range(0, len(converted_seqs), stepsize)):
            batched_sequences = converted_seqs[i:i+stepsize]
            data = [(f"protein{j}", seq) for j, seq in enumerate(batched_sequences)]
            _, _, batch_tokens = batch_converter(data)
            results = model(batch_tokens.to(dev), repr_layers=rep_lyr, return_contacts=False)
            representation = results['representations'][rep_lyr[0]].cpu().detach().numpy().mean(axis=1)
            rep.append(representation)
    print(f"Computed representations in shape (B, D) => {representation.shape}")
    rep_array = np.vstack(np.array(rep))
    return rep_array


def esm_routine(model_key: str, 
                data_key: str, 
                output_path: Path, 
                stepsize: int=25, 
                pseudo_lls: bool=False, 
                all_models: dict=AVAILABLE_MODELS
    ) -> None:
    device = torch.device("mps") # NOTE: for Mac M1 processor, otherwise "cuda"
    torch.cuda.empty_cache()
    # load model resources from external resources or cache
    model, alphabet = torch.hub.load("facebookresearch/esm:main", all_models.get(model_key))
    model.to(device)
    # load data
    seq_enc_alphabet = get_alphabet(data_key)
    batch_converter = alphabet.get_batch_converter()
    # wt_sequence, offset = get_wildtype_and_offset(data_key)
    sequences, _ = load_one_hot(data_key)
    converted_sequences = ["".join([list(seq_enc_alphabet.keys())[list(seq_enc_alphabet.values()).index(s_i)] for s_i in seq]) 
                            for seq in sequences]
    print(f"Loaded {data_key}: N={len(converted_sequences)}")
    esm_rep = compute_esm_representation(
            model=model, 
            converted_seqs=converted_sequences, 
            batch_converter=batch_converter,
            alphabet=alphabet,
            dev=device,
            stepsize=stepsize,
        )
    # persist results
    out_filepath = output_path / f'./{data_key}_{model_key}_rep.pkl'
    np.savez(out_filepath.with_suffix(".npz"), esm_rep)
    with open(out_filepath, "wb") as outfile:
        pickle.dump(esm_rep, outfile)
    if pseudo_lls:
        esm_pll = compute_esm_plls(
            model=model, 
            converted_seqs=converted_sequences, 
            batch_converter=batch_converter,
            alphabet=alphabet,
            dev=device,
            stepsize=stepsize,
        )
        out_filepath = output_path / f'./{data_key}_{model_key}_plls.pkl'
        np.savez(out_filepath.with_suffix(".npz"), esm_pll)
        with open(out_filepath, "wb") as outfile:
            pickle.dump(esm_pll, outfile)
    return


def main():
    parser = argparse.ArgumentParser(description="Experiment Specifications")
    parser.add_argument("-d", "--data", type=str, choices=ALL_DATASETS, help="Dataset identifier")
    parser.add_argument("-m", "--model", type=str, choices=AVAILABLE_MODELS.keys(), help="pLM model identifier")
    parser.add_argument("-s", "--stepsize", type=int, default=25, help="Compute in steps of '-s'")
    parser.add_argument("--pll", action="store_true", help="Compute Pseudo-Log-Likelihoods")
    args = parser.parse_args()
    # If no flags => Default behavior: iterate over all options
    if not args.data:
        data = ALL_DATASETS
    else:
        data = [args.data]
    if not args.model:
        model_keys = AVAILABLE_MODELS.keys()
    else:
        model_keys = [args.model]
    data_model_iterator = product(data, model_keys)
    for d_key, m_key in data_model_iterator:
        output_path = Path(__file__).parent.resolve() / "files"
        esm_routine(data_key=d_key, model_key=m_key, output_path=output_path, stepsize=args.stepsize, pseudo_lls=args.pll)


if __name__ == "__main__":
    main()
