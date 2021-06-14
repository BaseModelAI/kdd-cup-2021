import argparse
import pickle
import numpy as np
from ogb.lsc import MAG240MDataset
from root import ROOT
from ogb.lsc import MAG240MEvaluator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Working directory')
    parser.add_argument("--num-ensembles", type=int, default = 5, help='Number of ensembles')
    return parser



def get_final_predictions(dataset_idx, num_ensembles, working_dir):
    idx2output_all = {}
    
    for ensemble_id in range(num_ensembles):
        print(f"Predictions from {ensemble_id} ensemble")
        predictions = []
        with open(f'{working_dir}/test_ensemble_{ensemble_id}', 'rb') as handle:
            idx2output_ensemble = pickle.load(handle)
        if ensemble_id == 0:
            idx2output_all = idx2output_ensemble
        else:
            for k, v in idx2output_ensemble.items():
                idx2output_all[k] += v

        for idx in dataset_idx:
            prediction = idx2output_all[idx].argmax().item()
            predictions.append(prediction)

    return np.array(predictions)


def inference(params):
    mag_dataset = MAG240MDataset(ROOT)
    split_dict = mag_dataset.get_idx_split()
    test_idx = split_dict['test']
    predictions_test = get_final_predictions(test_idx, params.num_ensembles, params.working_dir)
    evaluator = MAG240MEvaluator()
    input_dict = {'y_pred': predictions_test}
    evaluator.save_test_submission(input_dict = input_dict, dir_path = '.')


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    inference(params)