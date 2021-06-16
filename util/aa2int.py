import numpy as np
from Bio.Seq import Seq


alphabet = "ARNDCQEGHILKMFPSTWYV-"
# map amino acids to integers (A->0, R->1, etc)
a2n = dict((a, n) for n, a in enumerate(alphabet))


def aa2int(x: str):
    return a2n.get(x, a2n['-'])


def seq2int(seq: str):
    int_seq = np.zeros(len(seq), dtype=np.int)
    for i, s in enumerate(seq):
        int_seq[i] = aa2int(s)
    return int_seq


def list_of_pairs_2_seq(list):
    seq_str = []
    last_i = -1
    for i, s in list:
        if i - last_i > 1:
            raise RuntimeError("Parsed Sequence has gaps!")
        last_i = i
        seq_str.append(s)
    return Seq(''.join(seq_str))


def map_alphabets(a_from: dict, a_to: dict):
    """
    Translates amino acids encoded with the from-alphabet into sequences with the other alphabet.
    :param a_from:
    :param a_to:
    :return:
    """
    for a in a_from.keys():
        if a not in a_to.keys():
            raise RuntimeError("The from-alphabet must be a subset of the to-alphabet!")
    return {a_from[a]: a_to[a] for a in a_from.keys()}


def map_is_identity(map: dict):
    """
    Checks if a map created with #map_alphabets is the identity.
    :param map:
    :return:
    """
    is_identity = True
    for i, j in map.items():
        is_identity = is_identity and i == j
        if not is_identity:
            break
    return is_identity
