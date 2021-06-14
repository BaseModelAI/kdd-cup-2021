import argparse
import numpy as np
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from ogb.lsc import MAG240MDataset
from root import ROOT


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Working directory')
    return parser


def sort_paper_cite_paper(working_dir):
     # edges_paper_cites_paper is sorted by first columns
    edges_paper_cites_paper = np.load(f'{ROOT}/mag240m_kddcup2021/processed/paper___cites___paper/edge_index.npy',  mmap_mode='r')

    # sort paper->cite->paper by second column
    print("Start sorting paper cite paper relation by second column")
    edges_paper_cites_paper_sorted_by_second_column = edges_paper_cites_paper.T[np.argsort(edges_paper_cites_paper.T[:, 1])].T
    np.save(os.path.join(working_dir, 'edges_paper_cites_paper_sorted_by_second_column.npy'), edges_paper_cites_paper_sorted_by_second_column)
    print("Done sorting")

def sort_author_paper(edge_author_paper, author_idxs, working_dir):
    edge_author_paper_sorted_by_paper = edge_author_paper[:,author_idxs].T[np.argsort(edge_author_paper.T[author_idxs, 1])]
    np.save(os.path.join(working_dir, 'edge_author_paper_sorted_by_paper.npy'), edge_author_paper_sorted_by_paper)


def get_labelled_idxs(dataset):
    """
    Get all indices with labels
    """
    split_dict = dataset.get_idx_split()
    train_idx = split_dict['train']
    valid_idx = split_dict['valid']
    test_idx = split_dict['test']
    idxs_labelled = set(train_idx)
    idxs_labelled = idxs_labelled.union(set(valid_idx))
    idxs_labelled = idxs_labelled.union(set(test_idx))
    return idxs_labelled


def find_authors_with_labels(edge_author_paper, idxs_labelled):
    autors_with_labelled_papers = set()
    edge_idx = []
    print("Finding all authors with associated labels")
    for i in tqdm(range(edge_author_paper.shape[1])):
        author = edge_author_paper[0,i]
        paper = edge_author_paper[1,i]
        if paper in idxs_labelled:
            autors_with_labelled_papers.add(author)
            edge_idx.append(i)
    return autors_with_labelled_papers, edge_idx


def find_another_papers_of_authors(edge_author_paper, autors_with_labelled_papers, working_dir, idxs_labelled):
    """
    Find all other papers of the authors of each paper with labels
    """

    author_to_papers = defaultdict(set)
    author_idxs = []
    print("Finding all other papers of the authors of each paper with labels")
    for i in tqdm(range(edge_author_paper.shape[1])):
        author = edge_author_paper[0,i]
        paper = edge_author_paper[1,i]
        if author in autors_with_labelled_papers:
            author_idxs.append(i)
            author_to_papers[author].add(paper)


    paper2thesameauthors_papers = defaultdict(list)
    for i in tqdm(range(edge_author_paper.shape[1])):
        author = edge_author_paper[0, i]
        paper = edge_author_paper[1, i]

        if paper in idxs_labelled:
            paper2thesameauthors_papers[paper] += author_to_papers[author]

    # remove current paper from value
    for paper, papers in paper2thesameauthors_papers.items():
        paper2thesameauthors_papers[paper] = list(filter(lambda x: x != paper, papers))

    with open(os.path.join(working_dir, 'paper2thesameauthors_papers'), 'wb') as handle:
        pickle.dump(paper2thesameauthors_papers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return author_idxs

def preprocess(working_dir):
    os.makedirs(working_dir, exist_ok=True)
    sort_paper_cite_paper(working_dir)

    dataset = MAG240MDataset(ROOT)
    idxs_labelled = get_labelled_idxs(dataset)

    edge_author_paper = np.load(f'{ROOT}/mag240m_kddcup2021/processed/author___writes___paper/edge_index.npy', mmap_mode='r')

    autors_with_labelled_papers, edge_idx = find_authors_with_labels(edge_author_paper, idxs_labelled)

    # author->paper edges but only for authors with labelled papers
    edge_author_paper_small = edge_author_paper[:,edge_idx]
    np.save(f'{working_dir}/edge_author_paper_small', edge_author_paper_small)

    author_idxs = find_another_papers_of_authors(edge_author_paper, autors_with_labelled_papers, params.working_dir, idxs_labelled)
    sort_author_paper(edge_author_paper, author_idxs,  working_dir)

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    preprocess(params.working_dir)