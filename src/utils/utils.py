import contextlib
import numpy as np
import random
import os

import torch
import pickle as pkl

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_metrics(path="saved/metrics.pkl", metrics=[]):
    with open(path, 'wb') as f:
        pkl.dump(metrics, f)


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    # filename = os.path.join(checkpoint_path, "self"+filename)
    # torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))
        torch.save(state,os.path.join(checkpoint_path, "model_best.pth"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)  
    model.load_state_dict(best_checkpoint["state_dict"])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copied from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

  

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
