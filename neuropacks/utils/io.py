import pickle


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))
