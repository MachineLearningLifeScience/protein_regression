"""
This code wraps up all code that can be found in the Notebooks:
    esm_representation.ipynb
    additional prot-T5 code
To be used to derive embeddings on cluster compute resources.
"""
from itertools import product
import sys
import argparse
from pathlib import Path
import numpy as np
from os.path import join
from tqdm import tqdm
import pickle
from typing import Tuple, List
import torch
from transformers import T5Tokenizer, T5EncoderModel
from data.load_dataset import get_wildtype_and_offset
from data.get_alphabet import get_alphabet
from data.load_dataset import load_one_hot


AVAILABLE_MODELS = {
    "esm1b": "esm1b_t33_650M_UR50S",
    "esm1v": "esm1v_t33_650M_UR90S_1",
    "esm2": "esm2_t36_3B_UR50D",
    "protT5": "prot_t5_xl_uniref50",
    }
ALL_DATASETS = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"]
DEVICES=["cuda", "mps", "cpu"]


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


def compute_esm_plls(model, seqs, seq_enc_alphabet, alphabet, dev, stepsize: int=10) -> np.ndarray:
    print("Compute PLL ...")
    batch_converter = alphabet.get_batch_converter()
    converted_sequences = ["".join([list(seq_enc_alphabet.keys())[list(seq_enc_alphabet.values()).index(s_i)] for s_i in seq]) 
                            for seq in seqs]
    log_ps = []
    with torch.no_grad():
        for i in tqdm(range(0, len(converted_sequences), stepsize)):
            batched_sequences = converted_sequences[i:i+stepsize]
            data = [(f"protein{j}", seq) for j, seq in enumerate(batched_sequences)]
            _, _, batch_tokens = batch_converter(data)
            log_probs = compute_pll(batched_sequences, model, batch_tokens, alphabet, device=dev)
        log_ps.append(log_probs)
    ll_array = np.vstack(np.array(log_ps))
    return ll_array


def compute_esm_representation(model, seqs: List[int], alphabet, seq_enc_alphabet: dict, dev, stepsize: int=10, rep_lyr: List[int]=[33]) -> np.ndarray:
    rep = []
    batch_converter = alphabet.get_batch_converter()
    converted_sequences = ["".join([list(seq_enc_alphabet.keys())[list(seq_enc_alphabet.values()).index(s_i)] for s_i in seq]) 
                            for seq in seqs]
    # Extract per-residue representations and limit memory requirements
    with torch.no_grad():
        for i in tqdm(range(0, len(converted_sequences), stepsize)):
            batched_sequences = converted_sequences[i:i+stepsize]
            data = [(f"protein{j}", seq) for j, seq in enumerate(batched_sequences)]
            _, _, batch_tokens = batch_converter(data)
            results = model(batch_tokens.to(dev), repr_layers=rep_lyr, return_contacts=False)
            representation = results['representations'][rep_lyr[0]].cpu().detach().numpy().mean(axis=1)
            rep.append(representation)
    print(f"Computed representations in shape (B, D) => {representation.shape}")
    rep_array = np.vstack(np.array(rep))
    return rep_array


def compute_prot_representation(model, seqs: List[str], alphabet, seq_enc_alphabet: dict, dev, stepsize: int=10, rep_lyr: List[int]=[33]) -> np.ndarray:
    rep = []
    # NOTE: ProtT5 requires space between elements
    converted_sequences = [" ".join([list(seq_enc_alphabet.keys())[list(seq_enc_alphabet.values()).index(s_i)] for s_i in seq]) 
                            for seq in seqs]
    # Extract per-residue representations and limit memory requirements
    with torch.no_grad():
        for i in tqdm(range(0, len(converted_sequences), stepsize)):
            batched_sequences = converted_sequences[i:i+stepsize]
            data = [(f"protein{j}", seq) for j, seq in enumerate(batched_sequences)]
            ids = alphabet(data, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids["input_ids"]).to(dev)
            attention_mask = torch.tensor(ids["attention_mask"]).to(dev)
            model_attn = model(input_ids=input_ids, attention_mask=attention_mask)
            representation = model_attn.cpu().detach().numpy().mean(axis=1)
            rep.append(representation)
    print(f"Computed representations in shape (B, D) => {representation.shape}")
    rep_array = np.vstack(np.array(rep))
    return rep_array


def load_model_from_repository(model_key, all_models=AVAILABLE_MODELS) -> tuple:
    if "esm" in model_key:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", all_models.get(model_key))
    elif "prot" in model_key:
        model = T5EncoderModel.from_pretrained(f"Rostlab/{all_models.get(model_key)}")
        alphabet = T5Tokenizer.from_pretrained(f"Rostlab/{all_models.get(model_key)}")
    else:
        raise ValueError(f"Specified model {model_key} not available!")
    return model, alphabet


def plm_routine(model_key: str, 
                data_key: str, 
                output_path: Path, 
                stepsize: int=25, 
                pseudo_lls: bool=False,
                device: str="cpu" 
    ) -> None:
    device = torch.device(device) # NOTE: for Mac M1 processor, otherwise "cuda"
    torch.cuda.empty_cache()
    # load model resources from external resources or cache
    model, alphabet = load_model_from_repository(model_key=model_key)
    model.to(device)
    # load data
    seq_enc_alphabet = get_alphabet(data_key)
    # wt_sequence, offset = get_wildtype_and_offset(data_key)
    sequences, _ = load_one_hot(data_key)
    print(f"Loaded {data_key}: N={len(sequences)}")
    if "esm" in model_key:
        plm_rep = compute_esm_representation(
                model=model, 
                seqs=sequences, 
                alphabet=alphabet,
                seq_enc_alphabet=seq_enc_alphabet,
                dev=device,
                stepsize=stepsize,
            )
    elif "prot" in model_key:
        plm_rep = compute_prot_representation(
                model=model, 
                seqs=sequences, 
                alphabet=alphabet,
                seq_enc_alphabet=seq_enc_alphabet,
                dev=device,
                stepsize=stepsize,
            )
    # persist results
    out_filepath = output_path / f'./{data_key}_{model_key}_rep.pkl'
    np.savez(out_filepath.with_suffix(".npz"), plm_rep)
    with open(out_filepath, "wb") as outfile:
        pickle.dump(plm_rep, outfile)
    if pseudo_lls:
        model_pll = compute_esm_plls(
            model=model, 
            seqs=sequences, 
            alphabet=alphabet,
            dev=device,
            stepsize=stepsize,
        )
        out_filepath = output_path / f'./{data_key}_{model_key}_plls.pkl'
        np.savez(out_filepath.with_suffix(".npz"), model_pll)
        with open(out_filepath, "wb") as outfile:
            pickle.dump(model_pll, outfile)
    return


def main():
    parser = argparse.ArgumentParser(description="Experiment Specifications")
    parser.add_argument("-d", "--data", type=str, choices=ALL_DATASETS, help="Dataset identifier")
    parser.add_argument("-m", "--model", type=str, choices=AVAILABLE_MODELS.keys(), help="pLM model identifier")
    parser.add_argument("-s", "--stepsize", type=int, default=25, help="Compute in steps of '-s'")
    parser.add_argument("--pll", action="store_true", help="Compute Pseudo-Log-Likelihoods")
    parser.add_argument("--device", type=str, default="cuda", choices=DEVICES)
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
        plm_routine(data_key=d_key, model_key=m_key, output_path=output_path, stepsize=args.stepsize, pseudo_lls=args.pll, device=args.device)


if __name__ == "__main__":
    main()
