import pickle


def info_to_BLOB(row):
    return pickle.dumps(row)


def BLOB_to_info(row):
    return pickle.loads(row)

