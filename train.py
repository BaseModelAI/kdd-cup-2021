import argparse
import pickle
import os
import torch
import gc
import time
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from root import ROOT
from ogb.lsc import MAG240MDataset
from sklearn.preprocessing import normalize
from model import Model
from trainer import KDDTrainer
from dataset import KddcupDataset



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Working directory')
    parser.add_argument("--sketch-dim-paper", type=int, default = 256, help='Sketch paper width')
    parser.add_argument("--sketch-depth-paper", type=int, default = 40, help='Sketch paper depth')
    parser.add_argument("--sketch-dim-institution", type=int, default = 128, help='Sketch institution width')
    parser.add_argument("--sketch-depth-institution", type=int, default = 40, help='Sketch institution depth')
    parser.add_argument("--num-ensembles", type=int, default = 5, help='Number of ensembles')
    parser.add_argument("--hidden-size", type=int, default = 3500, help='Model hidden size')
    parser.add_argument("--batch-size", type=int, default = 512, help='Training batch size')
    return parser



def predict(working_dir, loader, net, ensemble_id):
    start = time.time()
    idx2output = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            output = net(batch['input'].float().cuda()).softmax(dim=-1)
            idx2output.update(dict(zip([i.item() for i in batch['node_idx']], output.cpu().numpy())))
    print(f"Inference time {(time.time() - start)/60} minutes")
    with open(f'{working_dir}/ensemble_{ensemble_id}', 'wb') as handle:
        pickle.dump(idx2output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del idx2output



def train(params):
    working_dir = params.working_dir
    mag_dataset = MAG240MDataset(ROOT)
    split_dict = mag_dataset.get_idx_split()
    train_idx_all = split_dict['train']
    valid_idx = split_dict['valid']
    test_idx = split_dict['test']
    train_idx_all = np.concatenate((train_idx_all, valid_idx))

    print("Loading data")
    ORIG_DATA_DIR = os.path.join(ROOT, 'mag240m_kddcup2021/processed')
    node_label_all = np.load(f'{ROOT}mag240m_kddcup2021/processed/paper/node_label.npy')
    year = np.load(os.path.join(ORIG_DATA_DIR, 'paper/node_year.npy'))
    bert_features = np.load(os.path.join(ORIG_DATA_DIR, 'paper/node_feat.npy'), mmap_mode='r')
    edges_paper_cite_paper_sorted_by_first = np.load(os.path.join(ORIG_DATA_DIR, 'paper___cites___paper/edge_index.npy'), mmap_mode='r')

    edges_paper_cite_paper_sorted_by_second = np.load(os.path.join(working_dir, 'edges_paper_cites_paper_sorted_by_second_column.npy'), mmap_mode='r')

    codes_memmap_filename_bert = os.path.join(working_dir,'codes_bert_memmap')
    codes_memmap_bert = np.memmap(codes_memmap_filename_bert, dtype='uint8', mode='r', shape=(mag_dataset.num_papers, params.sketch_depth_paper))

    with open(os.path.join(working_dir, 'paper2thesameauthors_papers'), 'rb') as handle:
        paper2thesameauthors_papers = pickle.load(handle)


    with open(os.path.join(working_dir, 'paper2inst'), 'rb') as handle:
        paper2inst = pickle.load(handle)

    with open(os.path.join(working_dir, 'codes_inst2id'), 'rb') as handle:
        inds_codes_inst2id = pickle.load(handle)

    codes_inst = np.load(os.path.join(working_dir, 'inst_codes.npy'))
    edge_author_paper_sorted_by_paper = np.load(f'{working_dir}/edge_author_paper_sorted_by_paper.npy', mmap_mode='r')

    adj_t = torch.load(os.path.join(working_dir, 'adj.pt')).float()
    edge_author_paper = np.load(f'{working_dir}/edge_author_paper_small.npy', mmap_mode='r')
    print("Loading data finished")


    trainer = pl.Trainer(gpus=1,  max_epochs=1, logger=False, checkpoint_callback=False, num_sanity_val_steps=0)

    model_input_sketch_size = params.sketch_depth_paper * params.sketch_dim_paper * 3 + params.sketch_depth_institution * params.sketch_dim_institution

    LABEL_PROPAGATION_NUM_FEATURES = 29
    N_NUMERICAL_FEATURES = 16*21 # 21 continuous features - each represents as 16 scales of sin and cos. Check `dataset.encode_scalar_column` function

    model_input_size = model_input_sketch_size + bert_features.shape[1] + mag_dataset.num_classes * LABEL_PROPAGATION_NUM_FEATURES + N_NUMERICAL_FEATURES

    for ensemble_id in range(params.num_ensembles):
        start = time.time()
        print(f"Training {ensemble_id} ensemble")
        gc.collect()
        net = Model(model_input_size, params.hidden_size, mag_dataset.num_classes)
        steps_per_epoch = int(len(train_idx_all) / 2 / params.batch_size + 65)
        model = KDDTrainer(net, 1e-4, steps_per_epoch)
        np.random.shuffle(train_idx_all)

        # paper nodes for running Cleora
        train_idx_cleora = train_idx_all[:len(train_idx_all)//2]

        # paper nodes for EMDE
        train_idx_emde = train_idx_all[len(train_idx_all)//2:]

        # Training Cleora
        y_train = torch.from_numpy(mag_dataset.paper_label[train_idx_cleora]).to(torch.long)
        y_cleora_propagation = torch.zeros(mag_dataset.num_papers+mag_dataset.num_authors, mag_dataset.num_classes)

        # Initialize paper nodes with one hot encodded labels
        y_cleora_propagation[train_idx_cleora] = F.one_hot(y_train, mag_dataset.num_classes).float()

        idxs_set = set(train_idx_cleora)
        author2labels_current = defaultdict(lambda: np.zeros(mag_dataset.num_classes))

        # Initialize author nodes
        for i in range(edge_author_paper.shape[1]):
            author = edge_author_paper[0,i]
            paper = edge_author_paper[1,i]
            if paper in idxs_set:
                label = int(node_label_all[paper])
                author2labels_current[author][label] += 1

        for i, (author, labels) in enumerate(author2labels_current.items()):
            y_cleora_propagation[mag_dataset.num_papers+author] = torch.FloatTensor(normalize(labels[None], 'l2')[0])

        del author2labels_current
        gc.collect()

        print("Compute cleora")
        for i in range(2):
            y_cleora_propagation =  adj_t.matmul(y_cleora_propagation)
            if i == 0:
                print('normalize')
                y_cleora_propagation = F.normalize(y_cleora_propagation, p=2, dim=1)

        y_cleora_propagation = y_cleora_propagation.numpy()
        test_dataset = KddcupDataset(test_idx, bert_features, edge_author_paper_sorted_by_paper, edges_paper_cite_paper_sorted_by_first, edges_paper_cite_paper_sorted_by_second,
                        paper2thesameauthors_papers, codes_inst, inds_codes_inst2id, paper2inst, params.sketch_depth_institution, codes_memmap_bert, year, params.sketch_depth_paper, params.sketch_dim_paper,
                                    params.sketch_dim_institution, mag_dataset, node_label_all, y_cleora_propagation)

        test_loader = DataLoader(test_dataset, batch_size=128, num_workers=15, shuffle=False, drop_last=False)



        model.learning_rate = 1e-4
        np.random.shuffle(train_idx_emde)
        years_traininig = [year[i] for i in train_idx_emde]
        # sorting training examples by year
        train_idx_sorted = [x for _, x in sorted(zip(years_traininig, train_idx_emde), key=lambda pair: pair[0])]

        train_dataset = KddcupDataset(train_idx_sorted, bert_features, edge_author_paper_sorted_by_paper, edges_paper_cite_paper_sorted_by_first, edges_paper_cite_paper_sorted_by_second,
                        paper2thesameauthors_papers, codes_inst, inds_codes_inst2id, paper2inst, params.sketch_depth_institution, codes_memmap_bert, year, params.sketch_depth_paper, params.sketch_dim_paper,
                                    params.sketch_dim_institution, mag_dataset, node_label_all, y_cleora_propagation)

        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=15, shuffle=False, drop_last=False)
        trainer.fit(model, train_loader)


        model.learning_rate = 5e-6
        np.random.shuffle(train_idx_emde)
        years_traininig = [year[i] for i in train_idx_emde]
        train_idx_sorted = [x for _, x in sorted(zip(years_traininig, train_idx_emde), key=lambda pair: pair[0])]
        train_dataset = KddcupDataset(train_idx_sorted, bert_features, edge_author_paper_sorted_by_paper, edges_paper_cite_paper_sorted_by_first, edges_paper_cite_paper_sorted_by_second,
                        paper2thesameauthors_papers, codes_inst, inds_codes_inst2id, paper2inst, params.sketch_depth_institution, codes_memmap_bert, year, params.sketch_depth_paper, params.sketch_dim_paper,
                                    params.sketch_dim_institution, mag_dataset, node_label_all, y_cleora_propagation)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=15, shuffle=False, drop_last=False)
        trainer.fit(model, train_loader)

        net.eval()
        print(f"Training time {(time.time()-start) / 60} minutes")

        predict(working_dir, test_loader, net, ensemble_id)

        # save model
        batch = next(iter(train_loader))
        net.cpu()
        traced_script_module = torch.jit.trace(net, (batch['input'].float().cpu()))
        traced_script_module.save(os.path.join(working_dir, f'model_jit_{ensemble_id}'))

        np.save(f'{working_dir}/y_cleora_propagation_{ensemble_id}', y_cleora_propagation)
        del y_train
        del y_cleora_propagation



if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    train(params)