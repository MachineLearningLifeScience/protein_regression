__author__ = "R. Michael"
from pathlib import Path
from warnings import warn
import numpy as np
import pickle
from typing import List, Tuple
from Bio import SearchIO
from data.load_dataset import get_wildtype_and_offset
from data.get_alphabet import get_alphabet
from data.load_dataset import load_one_hot


"""
Using frequencies from UniProt, as provided by EMBL-EBI - see https://www.ebi.ac.uk/uniprot/TrEMBLstats
which is Release 2023_04 of 13-Sep-2023 of UniProtKB/TrEMBL contains 251600768 sequence entries,
comprising 88161770578 amino acids.

5.  AMINO ACID COMPOSITION
   5.1  Composition in percent for the complete database

   Ala (A) 9.03   Gln (Q) 3.80   Leu (L) 9.85   Ser (S) 6.82
   Arg (R) 5.84   Glu (E) 6.24   Lys (K) 4.93   Thr (T) 5.55
   Asn (N) 3.79   Gly (G) 7.27   Met (M) 2.33   Trp (W) 1.30
   Asp (D) 5.47   His (H) 2.22   Phe (F) 3.88   Tyr (Y) 2.88
   Cys (C) 1.29   Ile (I) 5.53   Pro (P) 4.99   Val (V) 6.86
"""
BACKGROUND_AA_FREQUENCIES = {
    "A": 0.0903,
    "C": 0.0129,
    "D": 0.0547,
    "E": 0.0624,
    "F": 0.0388,
    "G": 0.0727,
    "H": 0.0222,
    "I": 0.0553,
    "K": 0.0493,
    "L": 0.0985,
    "M": 0.0233,
    "N": 0.0379,
    "P": 0.0499,
    "Q": 0.0380,
    "R": 0.0584,
    "S": 0.0682,
    "T": 0.0555,
    "V": 0.0686,
    "W": 0.0130,
    "Y": 0.0288,
}

DATA_HMM_KVP = {
    "MTH3": "MTH3_HAEAESTABILIZED_1_b0.5",
    "TIMB": "TRPC_THEMA_1_b0.5",
    "CALM": "CALM1_HUMAN_1_b0.5",
    "1FQG": "BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105",
    "UBQT": "RL401_YEAST_1_b0.5",
    "BRCA": "BRCA1_HUMAN_BRCT_1_b0.3",
    "TOXI": "parEparD_3",
}


def read_hmm_emissions(hmm_file_path) -> Tuple[np.ndarray, List[str]]:
    """
    NOTE: This subroutine has largely been written by GPT-4 and was debugged and modified by the author.
    """
    with open(hmm_file_path, "r") as file:
        lines = file.readlines()

    # Find where the HMM matrix starts and ends
    start = next(i for i, line in enumerate(lines) if line.startswith("HMM "))
    end = next(
        i for i, line in enumerate(lines[start:], start) if line.startswith("//")
    )

    # Read the emission probabilities
    emissions = []
    aa_mapping = lines[start].split()[1:21]  # In what order do our AAs occur
    for line in lines[start + 2 : end]:  # Skip header lines
        parts = line.split()
        if not parts[
            0
        ].isnumeric():  # Only process lines starting with a match state index
            continue
        # Each match state has 20 emissions followed by 20 insert emissions; we'll take just the first 20
        match_emissions = list(map(float, parts[1:21]))
        emissions.append(match_emissions)
    emissions = np.array(emissions)
    return emissions, aa_mapping


def calculate_pssm(hmm_emissions, background_frequencies):
    """
    NOTE: This subroutine has largely been written by GPT-4
    """
    pssm = []
    for position_emissions in hmm_emissions:
        pssm_row = []
        for aa, emission in zip(
            background_frequencies.keys(), position_emissions
        ):  # NOTE: background frequencies have to be ordered
            # Calculate the log-odds score
            if emission > 0:
                score = np.log(emission / background_frequencies[aa])
            else:
                score = -np.inf  # Use negative infinity to represent log(0)
            pssm_row.append(score)
        pssm.append(pssm_row)
    pssm = np.array(pssm)
    return pssm


def compute_indices_from_alignment_against_sequence_wt_search(
    data_key: str,
) -> np.ndarray:
    """
    If sequences are of different length than the PSSM, the alignment and WT are different in length
    and subsequently the used HMM file.
    Use prior hmmsearch to obtain indices on which residues of the WT are part of the HMM/PSSM.
    """
    print(
        f"Length of WT {data_key} and PSSM different, aligning against HMMR search ..."
    )
    hmmsearch_filename = (
        Path(__file__).parent.resolve()
        / "files"
        / "hmms"
        / f"hmmsearch_{data_key.lower()}.txt"
    )
    hmmsearch = SearchIO.read(hmmsearch_filename, "hmmer3-text")
    if len(hmmsearch.hits[0].hsps) == 1:
        indices = np.arange(*hmmsearch.hits[0].hsps[0].hit_range)
    else:
        raise NotImplementedError(
            "Multiple Hits found! Index composition not yet implemented."
        )  # TODO
    return indices


def compute_sequence_encoding_from_pssm(
    data_key: str, pssm: np.ndarray, sequences: List[str], encoding_alphabet: dict
) -> np.ndarray:
    pssm_encoding = []
    # check data-dimensions against available PSSM
    if sequences.shape[1] != len(pssm):
        seq_indices = compute_indices_from_alignment_against_sequence_wt_search(
            data_key=data_key
        )
    else:
        seq_indices = np.arange(len(seq))
    if (
        seq_indices.shape[0] <= pssm.shape[0]
    ):  # Case: fewer hits from hmmsearch against WT than available from computed HMM - we have to subset
        warn("Less Sequence positions than PSSM positions!\n Subsetting PSSM ...")
        reset_search_idx = (
            seq_indices - seq_indices[0]
        )  # subtract starting position s.t. first element 0
        pssm = pssm[reset_search_idx]
    else:
        warn(
            "More Sequence positions than PSSM positions!\n Clipping Sequence index ..."
        )
        seq_indices = seq_indices[: pssm.shape[0]]
    encoding_aa_inverse_dict = {v: k for k, v in encoding_alphabet.items()}
    pssm_aa_lookup_list = list(BACKGROUND_AA_FREQUENCIES.keys())
    # assert BACKGROUND_AA_FREQUENCIES.keys() == encoding_alphabet.keys(), "Mismatch Sequence encoding and PSSM indices!"
    # compute PSSM scores for each sequence
    for seq in sequences:
        assert len(seq[seq_indices]) == len(
            pssm
        ), "Sequence and computed PSSM mismatch in length!"
        # NOTE: alphabets mismatch convert from seq label-encoding to PSSM encoding
        enc = np.array(
            [
                pssm[idx][pssm_aa_lookup_list.index(encoding_aa_inverse_dict[aa])]
                for idx, aa in enumerate(seq[seq_indices])
            ]
        )  # label encoded aa required to be the same as position in reference AA_FREQUENCIES KVP
        pssm_encoding.append(enc)
    pssm_encoding = np.stack(pssm_encoding)
    return pssm_encoding


def compute_pssm_and_persist(
    data_key: str,
    output_filepath: Path,
    hmm_filename_lookup=DATA_HMM_KVP,
    background_freq=BACKGROUND_AA_FREQUENCIES,
):
    hmm_identifier = hmm_filename_lookup.get(data_key)
    hmm_filename = output_filepath / "hmms" / f"{hmm_identifier}.hmm"

    hmm_emissions, hmm_aa_positions = read_hmm_emissions(hmm_filename)
    assert hmm_aa_positions == list(
        background_freq.keys()
    ), "AA mapping different between HMM and frequency dict!"
    pssm = calculate_pssm(hmm_emissions, background_freq)

    # load data
    seq_enc_alphabet = get_alphabet(data_key)
    # wt_sequence, offset = get_wildtype_and_offset(data_key)
    sequences, _ = load_one_hot(data_key)
    # encode sequences with computed PSSM
    encoding = compute_sequence_encoding_from_pssm(
        data_key=data_key,
        pssm=pssm,
        sequences=sequences,
        encoding_alphabet=seq_enc_alphabet,
    )

    # persist base PSSM
    if not (output_filepath / f"{data_key.lower()}_pssm.pkl").exists():
        with open(output_filepath / f"{data_key.lower()}_pssm.pkl", "wb") as outfile:
            pickle.dump(pssm, outfile)
    # persist encoding of sequences
    output_filename = output_filepath / f"{data_key.lower()}_pssm_rep.pkl"
    with open(output_filename, "wb") as outfile:
        pickle.dump(encoding, outfile)


if __name__ == "__main__":
    for data_key in DATA_HMM_KVP.keys():
        output_path = Path(__file__).parent.resolve() / "files"
        compute_pssm_and_persist(data_key, output_filepath=output_path)
