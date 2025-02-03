from dataclasses import dataclass, asdict
from typing import Literal, Union

import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sklearn.datasets
from scipy.io.arff import loadarff
import pandas as pd
from tqdm import tqdm
 
import ray
ray.init() 

SEED = 1729 
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
NUM_CPUS = 8

@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    boosting_type: Literal['gbdt', 'dart', 'rf'] = 'gbdt'
    num_leaves: int = 31
    max_depth: int = -1
    n_estimators: int = 300
    min_split_gain: float = 0.0
    min_child_weight: float = 1e-3
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = NUM_CPUS
    


def load_heart_statlog():
    data, metadata = loadarff('BNG_heart-statlog.arff')
    df = pd.DataFrame(data)
    column_types = {'age': int, 
                    'sex': [0,1], 
                    'chest': float, 
                    'resting_blood_pressure': float, 
                    'serum_cholestoral': float, 
                    'fasting_blood_sugar': [0,1], 
                    'resting_electrocardiographic_results': [0,1,2], 
                    'maximum_heart_rate_achieved': float, 
                    'exercise_induced_angina': [0,1], 
                    'oldpeak': float, 
                    'slope': int, 
                    'number_of_major_vessels': float, 
                    'thal': [3,6,7], 
                    'class': [b'present', b'absent']}
    feature_columns = ['age', 'sex', 'chest', 'resting_blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', 'slope', 'number_of_major_vessels', 'thal', ]
    label = df[['class']].map(lambda x: 1 if x == b'present' else 0)  # Note the double bracket, this keeps the column as a DataFrame
    data = df[feature_columns]
    categorical_features = [feature for feature in feature_columns if column_types[feature] not in (int, float)]
    data = data.astype({col: 'category' for col in categorical_features})
    #dataset = lgb.Dataset(data, label=label, free_raw_data=False, categorical_feature=categorical_features)
    return data, label
    
    
def load_data():
    X,y = load_heart_statlog()
    return X,y


def make_splits(X,y, rng: np.random.Generator):
    outer_seed = rng.integers(0, 2**32-1)
    outer_split = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    indices = np.arange(len(y))
    for modelling_indices, test_indices in outer_split.split(indices, y=y):
        inner_seed = rng.integers(0, 2**32-1)
        inner_split = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=inner_seed)
        modelling_y = y.iloc[modelling_indices]
        test_X = X.iloc[test_indices]
        test_y = y.iloc[test_indices]
        # N.b. splitting the indices here can lead to an easy to make mistake. The StratifiedKFold.split 
        # method will _reindex_ the input, and return splits of that index, but since the input is an 
        # index this is an easy thing to miss.
        for train_meta_indices, dev_meta_indices in inner_split.split(modelling_indices, y=modelling_y):  
            # This is where things can go wrong. The output from split is not a direct split 
            # of the ingoing modelling indices, instead it is a split of the reindex version 
            # of the input, so we need to index out the actual data point indices
            train_indices = modelling_indices[train_meta_indices]
            dev_indices = modelling_indices[dev_meta_indices]
            train_X = X.iloc[train_indices]
            train_y = y.iloc[train_indices]
            dev_X = X.iloc[dev_indices]
            dev_y = y.iloc[dev_indices]
            assert set(train_indices).isdisjoint(dev_indices) and set(train_indices).isdisjoint(test_indices), "The training set overlaps the dev or test set"
            yield {'train': {'X': train_X, 'y': train_y},
                   'dev': {'X': dev_X, 'y': dev_y},
                   'test': {'X': test_X, 'y': test_y}}
            

def train_model(dataset_split, training_config: TrainingConfig, seed=None):
    train_data = lgb.Dataset(dataset_split['train']['X'], label=dataset_split['train']['y'])
    dev_data = lgb.Dataset(dataset_split['dev']['X'], label=dataset_split['dev']['y'])
    test_X, test_y = dataset_split['test']['X'], dataset_split['test']['y']
    
    params = asdict(training_config)
    params['verbose'] = -1
    params['random_state'] = seed
    model = lgb.train(params, train_set=train_data, valid_sets=[dev_data], callbacks=[lgb.early_stopping(stopping_rounds=5)])
    predictions = model.predict(test_X)
    roc_auc = roc_auc_score(test_y, predictions)
    return roc_auc
    #model = lgb.train(params, train_set=train_data, valid_sets=[(dataset_split['dev']['X'], dataset_split['dev']['y'])])
    
@ray.remote(num_cpus=NUM_CPUS)
def process_work(work_package):
    roc_auc = train_model(work_package['split'], work_package['training_config'], work_package['seed'])
    return roc_auc

def main():
    training_config = TrainingConfig()
    
    X,y = load_data()
    
    rng = np.random.default_rng(SEED)
    work_packages = []
    for split in make_splits(X,y, rng):
        work_package = {'split': split, 'seed': rng.integers(0, 2**32-1), 'training_config': training_config}
        work_packages.append(work_package)
    
    eventual_performance_distribution = []
    for work_package in tqdm(work_packages, 'Work packages'):
        performance = process_work.remote(work_package)
        eventual_performance_distribution.append(performance)
    
    performance_distribution = ray.get(eventual_performance_distribution)
    alpha = 0.05
    phi_1, phi_2 = alpha/2, 1-alpha/2
    ci_low, median, ci_high = np.quantile(performance_distribution, (phi_1, 0.5, phi_2))
    print(f"Cross validated ROC AUC: {median} ({1-alpha:%}CI [{ci_low},{ci_high}])")
    
    

if __name__ == '__main__':
    main()