import pickle


def info_to_BLOB(info):
    return pickle.dumps(info)


def BLOB_to_info(BLOB):
    return pickle.loads(BLOB)

def cache_object(object, path):
    pickle.dumps(object)

