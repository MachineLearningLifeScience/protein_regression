"""
utility script to extract mean EVE embeddings from pretrained models
AUTHOR: Peter M Groth
DATE: 03/03/2023
"""
import argparse
import json
import os
import pickle
import sys

import pandas as pd
import torch
from Bio import SeqIO

from EVE.EVE import VAE_model


def generate_eve_latents(dataset: str, suffix: str, file_path: str="/home/pcq275/"):
    """Function to extract mean value of embedded protein sequence given trained EVE model

    Args:
        dataset: Name of dataset
        suffix: Suffix for trained model

    """
    model_parameters_location = f"EVE/EVE/default_model_params.json"
    VAE_checkpoint = f"{file_path}mnt/eve_checkpoint/{dataset}_{suffix}"
    output_dir = f"{file_path}protein_regression/data/files/embeddings/EVE_{dataset}_mean"
    output_erda = f"{file_path}mnt/eve_results/EVE_{dataset}_mean"
    os.makedirs(output_dir, exist_ok=True)
    dataset_pickle = f"data/interim/{dataset}/{dataset}_EVE_preprocessed.pkl"
    # Load dataframe
    df = pd.read_csv(f"data/interim/{dataset}/{dataset}.csv")
    names = df["name"].tolist()
    msa_path = f"data/raw/{dataset}/{dataset}_family.aln.fasta"

    # Load data
    with open(dataset_pickle, "rb") as handle:
        data = pickle.load(handle)

    invalid_sequences = []

    for name in names:
        if len(data.seq_name_to_sequence[">" + name]) == 0:
            print(f"Sequence {name} not found in processed dataset.")
            invalid_sequences.append(name)

    # Then, use the dictionary to extract sequences of interest.
    model_name = f"{dataset}_{suffix}"
    model_params = json.load(open(model_parameters_location))

    # Load model
    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42,
    )

    checkpoint_name = f"{VAE_checkpoint}_best"
    try:
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized VAE with checkpoint '{checkpoint_name}'.")
    except:
        print(f"Unable to locate VAE model checkpoint: {checkpoint_name}")
        sys.exit(0)

    # Send to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print("Moved model to GPU.")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate embeddings (latent representations/encodings)
    with torch.no_grad():
        # Iterate through global MSA file. Encode sequence if sequence in local set.
        fasta_sequences = SeqIO.parse(open(msa_path), "fasta")
        for fasta in fasta_sequences:
            name = fasta.id
            if name in names and name not in invalid_sequences:
                sequence = data.seq_name_to_sequence[f">{name}"]
                x = torch.zeros((data.seq_len, 20))
                for j, letter in enumerate(sequence):
                    if letter in data.aa_dict:
                        k = data.aa_dict[letter]
                        x[j, k] = 1.0
                if torch.cuda.is_available():
                    x = x.to(device)

                z, _ = model.encoder(x)
                output_path = f"{output_dir}/{name}.pt"
                torch.save(z.detach().cpu().clone(), output_path)


def main(dataset, suffix):
    generate_eve_latents(dataset, suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("suffix", type=str)
    args = parser.parse_args()
    main(args.dataset, args.suffix)
