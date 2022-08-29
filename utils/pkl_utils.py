import pickle


def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
