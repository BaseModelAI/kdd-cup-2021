import argparse
import random
import os
import math
import numpy as np
from tqdm import tqdm
from coders import get_vcoder
from root import ROOT


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Working directory')
    parser.add_argument("--sketch-dim", type=int, default = 256, help='Sketch width')
    parser.add_argument("--sketch-depth", type=int, default = 40, help='Sketch depth')
    return parser


def compute_sketches(params):
    os.makedirs(params.working_dir, exist_ok=True)

    # bert vectors
    features = np.load(f'{ROOT}/mag240m_kddcup2021/processed/paper/node_feat.npy', mmap_mode='r')
    
    N = 5_000_000
    idxs = random.sample(range(features.shape[0]), N)
    print(f"Getting {N} random nodes")
    random_features = features[idxs,:]

    vcoder = get_vcoder(random_features, params.sketch_depth, params.sketch_dim)

    codes_memmap_filename = os.path.join(params.working_dir, 'codes_bert_memmap')
    codes_memmap = np.memmap(codes_memmap_filename, dtype='uint8', mode='w+', shape=(features.shape[0], params.sketch_depth))

    print("Start computing codes")
    CHUNK_SIZE = 1_000_000
    for i in tqdm(range(math.ceil(features.shape[0] / CHUNK_SIZE))):
        emb = features[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE,:]
        codes = vcoder.transform(emb).astype(np.uint8)
        codes_memmap[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = codes


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    compute_sketches(params)