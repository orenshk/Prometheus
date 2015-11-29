import pickle


def load_classifier(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)