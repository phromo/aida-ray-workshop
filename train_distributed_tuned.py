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
from tqdm import tqdm, trange
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.train import SyncConfig
from ray.train import RunConfig

import ray
from ray import tune
ray.init() 


SEED = 1729 
N_OUTER_FOLDS = 3
N_INNER_FOLDS = 3
NUM_CPUS = 1
N_HPO_FOLDS = 1
DATASET_FILE = "BNG_heart-statlog_subsample.feather"

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
   
    
def load_data():
    df = pd.read_feather(DATASET_FILE)
    target_column = "class"
    feature_columns = [col for col in df.columns if col != target_column]
    X = df.loc[:, feature_columns]
    y = df[[target_column]]
    
    return X,y


def make_splits(indices, y, rng: np.random.Generator):
    outer_seed = rng.integers(0, 2**32-1)
    outer_splitter = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    
    outer_splits = dict()
    
    for outer_index, (modelling_meta_indices, test_meta_indices) in enumerate(outer_splitter.split(indices, y=y)):
        inner_seed = rng.integers(0, 2**32-1)
        inner_splitter = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=inner_seed)
        
        test_indices = indices[test_meta_indices]
        
        modelling_indices = indices[modelling_meta_indices]
        modelling_y = y.iloc[modelling_indices]
        inner_splits = dict()
        # N.b. splitting the indices here can lead to an easy to make mistake. The StratifiedKFold.split 
        # method will _reindex_ the input, and return splits of that index, but since the input is an 
        # index this is an easy thing to miss.
        for inner_index, (train_meta_indices, dev_meta_indices) in enumerate(inner_splitter.split(modelling_indices, y=modelling_y)):  
            # This is where things can go wrong. The output from split is not a direct split 
            # of the ingoing modelling indices, instead it is a split of the reindex version 
            # of the input, so we need to index out the actual data point indices
            train_indices = modelling_indices[train_meta_indices]
            dev_indices = modelling_indices[dev_meta_indices]
            
            assert set(train_indices).isdisjoint(dev_indices) and set(train_indices).isdisjoint(test_indices), "The training set overlaps the dev or test set"
            inner_splits[inner_index] =  {'train': train_indices, 'dev': dev_indices, 'test': test_indices}
        outer_splits[outer_index] = inner_splits
    
    return outer_splits

@ray.remote(num_cpus=NUM_CPUS)
def train_model(dataset_split, params, seed=None):
    X, y = load_data()
    train_X = X.iloc[dataset_split['train']]
    train_y = y.iloc[dataset_split['train']]
    dev_X = X.iloc[dataset_split['dev']]
    dev_y = y.iloc[dataset_split['dev']]
    test_X = X.iloc[dataset_split['test']]
    test_y = y.iloc[dataset_split['test']]
        
    train_data = lgb.Dataset(train_X, label=train_y)
    dev_data = lgb.Dataset(dev_X, label=dev_y)
    
    params['verbose'] = -1
    params['random_state'] = seed
    model = lgb.train(params, train_set=train_data, valid_sets=[dev_data], callbacks=[lgb.early_stopping(stopping_rounds=5)])
    predictions = model.predict(test_X)
    roc_auc = roc_auc_score(test_y, predictions)
    return roc_auc
    #model = lgb.train(params, train_set=train_data, valid_sets=[(dataset_split['dev']['X'], dataset_split['dev']['y'])])


def objective(config, dataset_splits):
    """Objective function for hyperparameter tuning in Ray Tune."""
    
    
    params = {
        "learning_rate": config["learning_rate"],
        "boosting_type": 'gbdt', #config["boosting_type"],
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "n_estimators": config["n_estimators"],
        "min_split_gain": config["min_split_gain"],
        "min_child_weight": config["min_child_weight"],
        "min_child_samples": config["min_child_samples"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "feature_fraction": config["feature_fraction"],
    }
    
    random_state = config["random_state"]
    rng = np.random.default_rng(random_state)
    
    fold_indices = rng.choice(list(dataset_splits.keys()), N_HPO_FOLDS)
    splits_to_train = {i: dataset_splits[i] for i in fold_indices}
    references = []

    for i, dataset_split in splits_to_train.items():
        # We should do smarter resplitting here, for now 
        # we just set the test split to be the same as the dev split
        # We're NOT allowed to use performance on the heldout set 
        # to determine hyper parameters
        resplit_dataset_split = {'train': dataset_split['train'], 
                                 'dev': dataset_split['dev'], 
                                 'test': dataset_split['dev']}  #Use performance on the dev set as HPO performance
        roc_auc_ref = train_model.remote(resplit_dataset_split, params, random_state)
        references.append(roc_auc_ref)
    
    # references is now a list of the object references for 
    # the inner nested cross validation training results
    performance_distribution = []
    unfinished = references
    
    while unfinished:
        # ray.wait takes the list of references (unfinished) and divides it 
        # into two list of the ones which are finished and the ones which aren't
        finished_tasks, unfinished = ray.wait(unfinished, num_returns=1)
        for finished_task in finished_tasks:
            # get the actual result from the reference
            performance = ray.get(finished_task)  
            performance_distribution.append(performance)
            
    median_roc_auc = np.median(performance)
    tune.report({"roc_auc": median_roc_auc})


@ray.remote
def process_outer_split(outer_fold_index, inner_splits, training_config, random_seed):
    results = []
    rng = np.random.default_rng(random_seed)
    
    # Define search space for hyperparameter tuning
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 0.1),
        "num_leaves": tune.randint(10, 100),
        "max_depth": tune.randint(3, 15),
        "n_estimators": tune.randint(50, 500),
        "min_split_gain": tune.uniform(0.0, 0.1),
        "min_child_weight": tune.loguniform(1e-5, 1e-1),
        "min_child_samples": tune.randint(5, 50),
        "reg_alpha": tune.loguniform(1e-4, 10),
        "reg_lambda": tune.loguniform(1e-4, 10),
        "random_state": tune.randint(0, 2**32-1),
        "feature_fraction": tune.uniform(0.6, 1.0),
    }

    # Define search algorithm and limit concurrency
    search_algo = OptunaSearch(metric="roc_auc", mode="max")
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=1)

    # Run hyperparameter tuning with Ray Tune
    num_trials = 1
    analysis = tune.run(
        tune.with_parameters(objective, dataset_splits=inner_splits),
        config=search_space,
        metric="roc_auc",
        mode="max",
        search_alg=search_algo,
        num_samples=num_trials,
        resources_per_trial=tune.PlacementGroupFactory([{'CPU': 4.0}] + [{'CPU': 1.0}] * num_trials),
    )

    # Get best hyperparameters
    best_config = analysis.get_best_config(metric="roc_auc", mode="max")

    for inner_index, split_indices in inner_splits.items():
        inner_seed = rng.integers(0, 2**32-1)
        # Train final model using best hyperparameters
        final_result = train_model.remote(split_indices, best_config, inner_seed)
        results.append(final_result)

    return results
 
    
    

def main():
    training_config = TrainingConfig()
    
    X,y = load_data()
    
    rng = np.random.default_rng(SEED)
    splits = make_splits(X.index, y, rng)

    
    # Remote tasks always return a reference to the result of that remote call
    # In the case of the outer loop that's a list of references for the nested 
    # remote calls it makes. We start by accumulating the  references to these 
    # lists (to not block starts of the outer loops)
    outer_loop_refs = []
    for outer_index, inner_splits in splits.items():
        split_seed = rng.integers(0,2**32-1)
        eventual_fold_results = process_outer_split.remote(outer_index, inner_splits, training_config, split_seed)
        # The process outer loop actually returns a reference to a list of references
        outer_loop_refs.append(eventual_fold_results)
    
    # Now that all the outer loop tasks have started, 
    # we collect their results - references to the inner loop tasks
    references = []
    for outer_loop_ref in outer_loop_refs:
        references.extend(ray.get(outer_loop_ref))

    # references is now a list of the object references for 
    # the inner nested cross validation training results
    performance_distribution = []
    unfinished = references
    
    while unfinished:
        # ray.wait takes the list of references (unfinished) and divides it 
        # into two list of the ones which are finished and the ones which aren't
        finished_tasks, unfinished = ray.wait(unfinished, num_returns=1)
        for finished_task in finished_tasks:
            # get the actual result from the reference
            performance = ray.get(finished_task)  
            performance_distribution.append(performance)
    
    alpha = 0.05
    phi_1, phi_2 = alpha/2, 1-alpha/2
    ci_low, median, ci_high = np.quantile(performance_distribution, (phi_1, 0.5, phi_2))
    print(f"Cross validated ROC AUC: {median} ({1-alpha:%}CI [{ci_low},{ci_high}])")
    
    

if __name__ == '__main__':
    main()