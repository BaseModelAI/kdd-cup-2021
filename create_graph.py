import argparse
import gc
import os
import numpy as np
import torch
from ogb.lsc import MAG240MDataset
from torch_sparse import SparseTensor
from root import ROOT


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Working directory')
    return parser


def create_graph(params):
    os.makedirs(params.working_dir, exist_ok=True)
    dataset = MAG240MDataset(ROOT)

    print("Preparing edges")
    edge_index_paper_cites_paper = dataset.edge_index('paper', 'cites', 'paper')
    edge_index_paper_cites_paper_symmetric = edge_index_paper_cites_paper[[1,0]]

    edge_index_autor_paper = dataset.edge_index('author', 'paper')
    # append `num_papers` to author id
    edge_index_autor_paper[0,:] = edge_index_autor_paper[0,:] + dataset.num_papers
    edge_index_autor_paper_symmetric = edge_index_autor_paper[[1,0]]

    edges = np.concatenate((edge_index_paper_cites_paper, edge_index_paper_cites_paper_symmetric,
                       edge_index_autor_paper,edge_index_autor_paper_symmetric), axis=1)

    del edge_index_autor_paper_symmetric
    del edge_index_autor_paper
    del edge_index_paper_cites_paper_symmetric
    del edge_index_paper_cites_paper
    gc.collect()

    print("Sorting edges - takes 45 mintues")
    edges = edges.T[np.argsort(edges.T[:, 0])].T
    edges = torch.from_numpy(edges)
    adj_t = SparseTensor(
        row=edges[0], col=edges[1],
        sparse_sizes=(dataset.num_papers+dataset.num_authors, dataset.num_papers+dataset.num_authors),
        is_sorted=True)

    torch.save(adj_t, os.path.join(params.working_dir, 'adj.pt'))


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    create_graph(params)