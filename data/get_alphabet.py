
# located here due to circular import between load_augmentation and load_dataset
def get_alphabet(name: str):
    if name.upper() in ["1FQG", "BLAT", "BRCA", "CALM", "MTH3", "TIMB", "UBQT", "TOXI"]:
        data_alphabet = list(enumerate([
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
        data_alphabet = {a: i for (i, a) in data_alphabet}
        return data_alphabet
    raise ValueError("Unknown dataset: %s" % name)


def get_eve_alphabet():
    """
    Alphabet as used in EVE data preprocessing.
    See https://github.com/OATML-Markslab/EVE/blob/460d70efeeeded58bc69227a203540d68953ae88/utils/data_utils.py#L44
    """
    return "ACDEFGHIKLMNPQRSTVWY"