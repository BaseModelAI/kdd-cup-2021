# Synerise at KDD Cup 2021: The paper citation challenge

Implementation of our solution to [KDD CUP Challenge](https://ogb.stanford.edu/kddcup2021/mag240m). The goal of the challenge is to predict the subject areas of papers situated in the heterogeneous graph in MAG240M-LSC dataset.

## Requirements
* Python 3.8
* Install requirments: `pip install -r requirements.txt`
* GPU for training
* SSD drive for fast reading memmap files
* 400 GB RAM
* Download binary [Cleora release](https://github.com/Synerise/cleora/releases/download/v1.1.0/cleora-v1.1.0-x86_64-unknown-linux-gnu). Then add execution permission to run it. Refer to [cleora github webpage](https://github.com/Synerise/cleora) for more details about Cleora.

## Getting Started
Steps 1-4 can be run simultaneously

1. Data preparation. The `MAG240M-LSC` dataset will be automatically downloaded if not exists to the path denoted in `root.py`. This takes a while (several hours to a day) in the first run, so please be patient. After decompression, the file size will be around 202GB.
Please change its content accordingly if you want to download the dataset to a custom hard-drive or folder. 
This script creates preprocessed data that is used then during training:
    * `data/edges_paper_cites_paper_sorted_by_second_column.npy` - numpy array with paper->cite->paper edges sorted by cited paper. 
    Used then for fast retrieval of the papers that cited selected paper.
    * `data/edge_author_paper_sorted_by_paper.npy` - numpy array with author->writes->paper sorted by paper.  Used then for fast retrieval of all paper authors.
    * `data/paper2thesameauthors_papers` - pickled dict that contains all other papers of the same authors as selected paper
    * `data/edge_author_paper_small` - edges author->paper but only for authors with labelled papers (for faster searching during training)

    ```
    python preprocessing.py
    ```
    Estimated time of preprocessing, without downloading data: 60 minutes

2. Compute paper sketches from bert features using [EMDE](https://arxiv.org/abs/2006.01894)
    ```
    python compute_paper_sketches.py
    ```
    It creates memmap file with paper sketches:
    `data/codes_bert_memmap`

    Estimated time of computing paper sketches: 105 minutes

3. Compute institutions sketches using [Cleora](https://github.com/Synerise/cleora) and [EMDE](https://arxiv.org/abs/2006.01894) 
    ```
    python compute_institutions_sketches.py
    ```
    It creates:
    * `data/inst_codes.npy` - memmap file with institutions sketches
    * `data/paper2inst` - pickled dict that contains all institutions for given paper
    * `data/codes_inst2id` - pickled dict that maps institution to its index in `data/inst_codes.npy`

    Estimated time of computing institutions sketches: 55 minutes

4. Create adjency matrix of graph with paper and author nodes
    ```
    python create_graph.py
    ```
    It creates `data/adj.pt` file that represents sparse adjency matrix

    Estimated time: 60 minutes

5. Training model for 2 epochs
    ```
    python train.py
    ```
    Final model was trained with 60 ensembles ```python train.py --num-ensembles 60```

    Predictions for test set for each ensemble are saved as `data/ensemble_{ensemble_id}`

    Two epochs training time: 40 minutes per one ensemble on Tesla V100 GPU
    
    Inference time for all test data: 7 minutes

6. Merging ensemble predictions and save the test submission to file `y_pred_mag240m.npz`
    ```
    python inference.py
    ```
