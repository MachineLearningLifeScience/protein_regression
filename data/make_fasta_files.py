"""
Utility script to generate .fasta files from loaded reference data.
"""

from pathlib import Path
from data.load_dataset import get_wildtype_and_offset
from data.get_alphabet import get_alphabet


ALL_DATASETS = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA", "TOXI"]


if __name__ == "__main__":
    for data_key in ALL_DATASETS:
        # load data
        seq_enc_alphabet = get_alphabet(data_key)
        wt, offset = get_wildtype_and_offset(data_key)
        output_path = (
            Path(__file__).parent.resolve()
            / "files"
            / "fastas"
            / f"{data_key}_wt.fasta"
        )
        converted_wt_sequence = "".join(
            [
                list(seq_enc_alphabet.keys())[
                    list(seq_enc_alphabet.values()).index(s_i)
                ]
                for s_i in wt
            ]
        )
        # persist results
        with open(output_path, "w") as fastafile:
            fastafile.write(f"> Seq_WT_{data_key}\n{converted_wt_sequence}")
