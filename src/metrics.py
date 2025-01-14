import pandas as pd
import numpy as np
import json
import zarr

PARTICLE= [
    {
        "name": "apo-ferritin",
        "difficulty": 'easy',
        "pdb_id": "4V1W",
        "label": 1,
        "color": [0, 255, 0, 0],
        "radius": 60,
        "map_threshold": 0.0418
    },
    {
        "name": "beta-amylase",
        "difficulty": 'ignore',
        "pdb_id": "1FA2",
        "label": 2,
        "color": [0, 0, 255, 255],
        "radius": 65,
        "map_threshold": 0.035
    },
    {
        "name": "beta-galactosidase",
        "difficulty": 'hard',
        "pdb_id": "6X1Q",
        "label": 3,
        "color": [0, 255, 0, 255],
        "radius": 90,
        "map_threshold": 0.0578
    },
    {
        "name": "ribosome",
        "difficulty": 'easy',
        "pdb_id": "6EK0",
        "label": 4,
        "color": [0, 0, 255, 0],
        "radius": 150,
        "map_threshold": 0.0374
    },
    {
        "name": "thyroglobulin",
        "difficulty": 'hard',
        "pdb_id": "6SCJ",
        "label": 5,
        "color": [0, 255, 255, 0],
        "radius": 130,
        "map_threshold": 0.0278
    },
    {
        "name": "virus-like-particle",
        "difficulty": 'easy',
        "pdb_id": "6N4V",
        "label": 6,
        "color": [0, 0, 0, 255],
        "radius": 135,
        "map_threshold": 0.201
    }
]

PARTICLE_COLOR=[[0,0,0]]+[
    PARTICLE[i]['color'][1:] for i in range(6)
]
PARTICLE_NAME=['none']+[
    PARTICLE[i]['name'] for i in range(6)
]

'''
(184, 630, 630)  
(92, 315, 315)  
(46, 158, 158)  
'''

def read_one_data(id, static_dir):
    zarr_dir = f'{static_dir}/{id}/VoxelSpacing10.000'
    zarr_file = f'{zarr_dir}/denoised.zarr'
    zarr_data = zarr.open(zarr_file, mode='r')
    volume = zarr_data[0][:]
    max = volume.max()
    min = volume.min()
    volume = (volume - min) / (max - min)
    volume = volume.astype(np.float16)
    return volume


def read_one_truth(id, overlay_dir):
    location={}

    json_dir = f'{overlay_dir}/{id}/Picks'
    for p in PARTICLE_NAME[1:]:
        json_file = f'{json_dir}/{p}.json'

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        num_point = len(json_data['points'])
        loc = np.array([list(json_data['points'][i]['location'].values()) for i in range(num_point)])
        location[p] = loc

    return location

def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


class ParticleEvaluator:
    def __init__(self, overlay_dir):
        self.overlay_dir = overlay_dir

    def compute_metrics(self, predictions, experiment_ids):
        eval_df = []
        for id in experiment_ids:
            truth = read_one_truth(id, self.overlay_dir)
            id_preds = predictions[predictions['experiment'] == id]
            for p in PARTICLE:
                xyz_truth = truth[p.name]
                xyz_predict = id_preds[id_preds['particle_type'] == p.name][['x', 'y', 'z']].values
                hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius * 0.5)
                eval_df.append({
                    'id': id,
                    'particle_type': p.name,
                    'P': metric[0],
                    'T': metric[1],
                    'hit': metric[2],
                    'miss': metric[3],
                    'fp': metric[4],
                })
        
        eval_df = pd.DataFrame(eval_df)
        gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
        gb['precision'] = gb['hit'] / gb['P'].fillna(0)
        gb['recall'] = gb['hit'] / gb['T'].fillna(0)
        gb['f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall']).fillna(0)

        gb = gb.sort_values('particle_type').reset_index(drop=False)
        gb['weight'] = [1, 0, 2, 1, 2, 1]  # 適切な重みを設定
        lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()

        return gb, lb_score



"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd

from scipy.spatial import KDTree


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta







## 学習時のメトリクス
import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        y_pred = np.argmax(y_pred, axis=1)
        
        dice = self.mean_dice_coefficient(y_true, y_pred)
        iou = self.mean_iou(y_true, y_pred)
        accuracy = self.pixel_accuracy(y_true, y_pred)
        precision, recall = self.precision_recall(y_true, y_pred)

        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    def mean_dice_coefficient(self, y_true, y_pred):
        dice_scores = []
        for class_idx in range(self.num_classes):
            y_true_class = (y_true == class_idx).astype(int)
            y_pred_class = (y_pred == class_idx).astype(int)
            dice_scores.append(self.dice_coefficient(y_true_class, y_pred_class))
        return np.mean(dice_scores)

    def mean_iou(self, y_true, y_pred):
        iou_scores = []
        for class_idx in range(self.num_classes):
            y_true_class = (y_true == class_idx).astype(int)
            y_pred_class = (y_pred == class_idx).astype(int)
            iou_scores.append(self.iou_score(y_true_class, y_pred_class))
        return np.mean(iou_scores)

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

    @staticmethod
    def iou_score(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return intersection / (union + 1e-7)

    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision_recall(y_true, y_pred):
        tp = np.sum((y_true == y_pred) & (y_pred != 0))
        fp = np.sum((y_true != y_pred) & (y_pred != 0))
        fn = np.sum((y_true != y_pred) & (y_true != 0))
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return precision, recall
