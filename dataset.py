import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from collections import Counter


def multiscale(x, scales):
    return np.hstack([x.reshape(-1,1)/pow(2., i) for i in scales])


def encode_scalar_column(x, scales=[-1, 0, 1, 2, 3, 4, 5, 6]):
    return np.hstack([np.sin(multiscale(x, scales)), np.cos(multiscale(x, scales))])


def find_paper_authors(paper_id, edge_author_paper_sorted_by_paper):
    start_id = np.searchsorted(edge_author_paper_sorted_by_paper[:,1], paper_id, side='left')
    end_id = np.searchsorted(edge_author_paper_sorted_by_paper[:,1], paper_id, side='right')
    authors = [edge_author_paper_sorted_by_paper[i,0] for i in range(start_id, end_id)]
    return authors



class KddcupDataset(Dataset):
    def __init__(self, idxs, bert_features, edge_author_paper_sorted_by_paper, edges_paper_cite_paper_sorted_by_first, edges_paper_cite_paper_sorted_by_second, 
                paper2thesameauthors_papers, codes_inst, codes_inst2id, paper2inst, n_codes_inst, codes_memmap_bert, year, n_codes_bert,
                sketch_dim_paper, sketch_dim_institution, dataset, node_label_all, y_cleora_propagation):
        self.idxs = idxs
        self.bert_features = bert_features
        self.edge_author_paper_sorted_by_paper = edge_author_paper_sorted_by_paper
        self.edges_paper_cite_paper_sorted_by_first = edges_paper_cite_paper_sorted_by_first
        self.edges_paper_cite_paper_sorted_by_second = edges_paper_cite_paper_sorted_by_second
        self.paper2thesameauthors_papers = paper2thesameauthors_papers
        self.codes_inst = codes_inst
        self.codes_inst2id = codes_inst2id
        self.paper2inst = paper2inst
        self.n_codes_inst = n_codes_inst
        self.codes_memmap_bert = codes_memmap_bert
        self.year = year
        self.n_codes_bert = n_codes_bert
        self.sketch_dim = sketch_dim_paper
        self.sketch_dim_institution = sketch_dim_institution
        self.dataset = dataset
        self.node_label_all = node_label_all
        self.y_cleora_propagation = y_cleora_propagation

    def __len__(self):
        return len(self.idxs)

    def create_sketch(self, example_idx, codes_array, n_codes):
        sketch = np.zeros(n_codes*self.sketch_dim)
        codes = codes_array[example_idx, :]
        assert np.sum(codes) > 0 # check if not empty codes
        for i, c in enumerate(codes):
            sketch[c + i*self.sketch_dim] = 1
        return sketch

    def create_sketch_summary_dst(self, n_codes, edges_to, codes_array):
        codes_edges = codes_array[[n for n in edges_to], :n_codes]
        assert 0 not in np.sum(codes_edges, axis=1) # check if not empty codes
        sketch_cited = np.zeros(n_codes*self.sketch_dim)
        for i_code in range(n_codes):
            np.add.at(sketch_cited,  codes_edges[:,i_code] + self.sketch_dim*i_code, 1)
        return normalize(sketch_cited.reshape(-1, self.sketch_dim), 'l2').reshape((n_codes*self.sketch_dim,))

    def calculate_labels_vector(self, edges_around, node_year, last_n=None):
        labels_around = np.zeros(self.dataset.num_classes)
        labels = self.node_label_all[edges_around]

        labels_years = self.year[edges_around]

        if last_n:
            labels = np.array([l for i,l in enumerate(labels) if labels_years[i] < node_year and labels_years[i] >= node_year-last_n])
        else:
            labels = np.array([l for i,l in enumerate(labels) if labels_years[i] < node_year])

        labels = labels[~np.isnan(labels)]
        labels = labels[labels!=-1]
        np.add.at(labels_around, labels.astype(int), 1)
        labels_norm = np.linalg.norm(labels_around, 2)
        labels_around = normalize(labels_around[None], 'l2')[0]
        return labels_around, labels_norm

    def calculate_labels_vector_weights(self, edges_around, node_year, weights):
        labels_around = np.zeros(self.dataset.num_classes)

        labels = self.node_label_all[edges_around]
        if len(labels) > 0:
            labels_years = self.year[edges_around]
            labels = np.array([l for i,l in enumerate(labels) if labels_years[i] < node_year])
            weights = np.array([weights[i] for i,l in enumerate(labels) if labels_years[i] < node_year])

            weights = weights[~np.isnan(labels)]
            labels = labels[~np.isnan(labels)]

            weights = weights[labels!=-1]
            labels = labels[labels!=-1]

            np.add.at(labels_around, labels.astype(int), weights)

        labels_norm = np.linalg.norm(labels_around, 2)
        labels_around = normalize(labels_around[None], 'l2')[0]
        return labels_around, labels_norm

    def calculate_cleora_labels_vector(self, cleora_vec, edges_around, node_year, current_year=False):
        labels_around = np.zeros(self.dataset.num_classes)
        if current_year:
            edges_prev = [p for p in edges_around if self.year[p] == node_year]
        else:
            edges_prev = [p for p in edges_around if self.year[p] < node_year]
        if edges_prev:
            labels_around = np.sum(normalize(cleora_vec[edges_prev], 'l2'), axis=0)
        return normalize(labels_around[None], 'l2')[0]

    def __getitem__(self, idx):
        node_idx = self.idxs[idx]

        bert_embedding = self.bert_features[node_idx]

        # find edges to cited papers
        start_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], node_idx, side='left')
        end_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], node_idx, side='right')
        edges_to = [self.edges_paper_cite_paper_sorted_by_first[1,i] for i in range(start_id, end_id)]
        assert not node_idx in edges_to

        sketch_cited_bert = self.create_sketch_summary_dst(self.n_codes_bert, edges_to, self.codes_memmap_bert)
        labels_around, labels_norm = self.calculate_labels_vector(edges_to, self.year[node_idx])
        labels_around_set, labels_norm_set = self.calculate_labels_vector(list(set(edges_to)), self.year[node_idx])
        labels_around_last2, labels_norm_last2 = self.calculate_labels_vector(edges_to, self.year[node_idx], 2)
        labels_around_last1, labels_norm_last1 = self.calculate_labels_vector(edges_to, self.year[node_idx], 1)

        edges_to_cleora_y = self.calculate_cleora_labels_vector(self.y_cleora_propagation, edges_to, self.year[node_idx])
        edges_to_cleora_y_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, edges_to, self.year[node_idx], current_year=True)


        # 2nd hop edged to
        edges_to_2nd = []
        edges_to_set =set(edges_to)
        for e in set(edges_to):
            start_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], e, side='left')
            end_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], e, side='right')
            candidates =  [self.edges_paper_cite_paper_sorted_by_first[1,i] for i in range(start_id, end_id)]
            candidates = [c for c in candidates if c != node_idx and c not in edges_to_set]
            edges_to_2nd += candidates
        assert not node_idx in edges_to_2nd
        labels_around_2nd, labels_norm_2nd = self.calculate_labels_vector(edges_to_2nd, self.year[node_idx])
        edges_to_cleora_y_2nd = self.calculate_cleora_labels_vector(self.y_cleora_propagation, edges_to_2nd, self.year[node_idx])
        edges_to_cleora_y_2nd_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, edges_to_2nd, self.year[node_idx], current_year=True)



        another_paper_edges_to = []
        for e in edges_to:
            another_paper_edges_to += self.paper2thesameauthors_papers[e]
        another_paper_edges_to = [a for a in another_paper_edges_to if self.year[a] < self.year[node_idx] and a != node_idx ]
        another_paper_edges_to_counter =  Counter(another_paper_edges_to).most_common(100)
        another_paper_edges_to_most_commons = [a[0] for a in another_paper_edges_to_counter]
        weights =  [a[1] for a in another_paper_edges_to_counter]

        assert not node_idx in another_paper_edges_to_most_commons
        another_paper_edges_to_labels, another_paper_edges_to_norm = self.calculate_labels_vector_weights(another_paper_edges_to_most_commons, self.year[node_idx], weights)
        another_paper_edges_to_y = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_paper_edges_to_most_commons, self.year[node_idx])
        another_paper_edges_to_y_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_paper_edges_to_most_commons, self.year[node_idx], current_year=True)


        # find edges to papers that cite this paper
        start_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_second[1], node_idx, side='left')
        end_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_second[1], node_idx, side='right')
        edges_from = [self.edges_paper_cite_paper_sorted_by_second[0,i] for i in range(start_id, end_id)]
        assert not node_idx in edges_from
        sketch_from_bert = self.create_sketch_summary_dst(self.n_codes_bert, edges_from, self.codes_memmap_bert)
        labels_being_cited, labels_being_cited_norm = self.calculate_labels_vector(edges_from, self.year[node_idx])
        another_paper_edges_from_y_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, edges_from, self.year[node_idx], current_year=True)



        another_papers = self.paper2thesameauthors_papers[node_idx]
        assert not node_idx in another_papers
        another_papers_labels, another_papers_labels_norm = self.calculate_labels_vector(another_papers, self.year[node_idx])
        another_papers_labels_set, another_papers_labels_norm_set = self.calculate_labels_vector(list(set(another_papers)), self.year[node_idx])
        another_papers_labels_last2, another_papers_labels_norm_last2 = self.calculate_labels_vector(another_papers, self.year[node_idx], 2)
        another_papers_labels_last1, another_papers_labels_norm_last1 = self.calculate_labels_vector(another_papers, self.year[node_idx], 1)
        sketch_another_papers_bert = self.create_sketch_summary_dst(self.n_codes_bert, another_papers, self.codes_memmap_bert)
        another_papers_cleora_y = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers, self.year[node_idx])
        another_papers_cleora_y_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers, self.year[node_idx], current_year=True)



        another_papers_2nd = []
        another_papers_older = [a for a in another_papers]
        another_papers_most_counter = Counter(another_papers_older).most_common(10)
        another_papers_most_commons = [a[0] for a in another_papers_most_counter]

        for p in another_papers_most_commons:
            another_papers_2nd += self.paper2thesameauthors_papers[p]

        another_papers_2nd_counter = Counter(another_papers_2nd).most_common(300)
        another_papers_2nd = [a[0] for a in another_papers_2nd_counter]

        another_papers_set = set(another_papers)
        another_papers_2nd = [p for p in another_papers_2nd if p not in another_papers_set and node_idx != p]
        assert not node_idx in another_papers_2nd
        another_papers_2nd_labels, another_papers_2nd_labels_norm = self.calculate_labels_vector(another_papers_2nd, self.year[node_idx])

        another_papers_cleora_y_2nd = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers_2nd, self.year[node_idx])
        another_papers_cleora_y_2nd_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers_2nd, self.year[node_idx], current_year=True)



        num_institutions = len(set(self.paper2inst[node_idx]))
        codes_edges = self.codes_inst[[self.codes_inst2id[int(n)] for n in self.paper2inst[node_idx] if int(n) in self.codes_inst2id]]
        sketch_inst = np.zeros(self.sketch_dim_institution*self.n_codes_inst)
        for i_code in range(self.n_codes_inst):
            np.add.at(sketch_inst,  codes_edges[:,i_code] + self.sketch_dim_institution*i_code, 1)

        sketch_inst = normalize(sketch_inst.reshape(-1, self.sketch_dim_institution), 'l2').reshape((self.n_codes_inst*self.sketch_dim_institution,))


        label_propagation = self.y_cleora_propagation[node_idx]
        label_propagation_norm = np.linalg.norm(label_propagation, 2)
        label_propagation_l1 = normalize(label_propagation[None], 'l1')[0]
        label_propagation = normalize(label_propagation[None], 'l2')[0]

        paper_authors_all = find_paper_authors(node_idx, self.edge_author_paper_sorted_by_paper)

        labels_authors_propagation = np.sum(normalize(self.y_cleora_propagation[[k + self.dataset.num_papers for k in paper_authors_all]], 'l2'), axis=0)
        labels_authors_propagation_norm = np.linalg.norm(labels_authors_propagation, 2)
        labels_authors_propagation = normalize(labels_authors_propagation[None], 'l2')[0]


        another_papers_citing = []
        for e in set(another_papers_most_commons):
            start_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], e, side='left')
            end_id = np.searchsorted(self.edges_paper_cite_paper_sorted_by_first[0], e, side='right')
            candidates =  [self.edges_paper_cite_paper_sorted_by_first[1,i] for i in range(start_id, end_id)]
            candidates = [c for c in candidates if c != node_idx]
            another_papers_citing += candidates
        assert not node_idx in another_papers_citing
        another_papers_citing_labels, another_papers_citing_labels_norm = self.calculate_labels_vector(another_papers_citing, self.year[node_idx])
        another_papers_citing_cleora_y = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers_citing, self.year[node_idx])
        another_papers_citing_cleora_y_current_year = self.calculate_cleora_labels_vector(self.y_cleora_propagation, another_papers_citing, self.year[node_idx], current_year=True)



        features = encode_scalar_column(np.array([len(another_papers),
                                                  len(edges_to),
                                                  len(edges_from),
                                                  labels_norm,
                                                  another_papers_labels_norm,
                                                  labels_being_cited_norm,
                                                  labels_norm_last2,
                                                  labels_norm_last1,
                                                  another_papers_labels_norm_last2,
                                                  another_papers_labels_norm_last1,
                                                  labels_norm_set,
                                                  another_papers_labels_norm_set,
                                                  labels_authors_propagation_norm,
                                                  label_propagation_norm,
                                                  labels_norm_2nd,
                                                  another_paper_edges_to_norm,
                                                  another_papers_2nd_labels_norm,
                                                  another_papers_citing_labels_norm,
                                                  num_institutions,
                                                  len(paper_authors_all),
                                                  self.year[node_idx]])).flatten()

        return {
             'input': np.concatenate((sketch_cited_bert,
                                      sketch_from_bert,
                                      sketch_another_papers_bert,
                                      bert_embedding,
                                      labels_around,
                                      another_papers_labels,
                                      features,
                                      sketch_inst,
                                      labels_being_cited,
                                      labels_around_last2,
                                      labels_around_last1,
                                      another_papers_labels_last2,
                                      another_papers_labels_last1,
                                      labels_around_set,
                                      another_papers_labels_set,
                                      labels_around_2nd,
                                      another_papers_2nd_labels,
                                      another_paper_edges_to_labels,
                                      another_paper_edges_from_y_current_year,
                                      label_propagation,
                                      labels_authors_propagation,
                                      another_papers_cleora_y,
                                      edges_to_cleora_y,
                                      edges_to_cleora_y_2nd,
                                      another_paper_edges_to_y,
                                      another_papers_cleora_y_2nd,
                                      edges_to_cleora_y_current_year,
                                      edges_to_cleora_y_2nd_current_year,
                                      another_paper_edges_to_y_current_year,
                                      another_papers_cleora_y_current_year,
                                      another_papers_citing_labels,
                                      another_papers_citing_cleora_y,
                                      another_papers_citing_cleora_y_current_year,
                                      another_papers_cleora_y_2nd_year,
                                      label_propagation_l1
                                      )),
             'target': self.node_label_all[node_idx],
             'node_idx': node_idx
        }
