import numpy as np
import pandas as pd
from src.LieTorch.tensor.tensor import Tensor

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class DataLoader():
    def __init__(self, dataset, batch_size=32, shuffle=False, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.X = dataset.X
        self.y = dataset.y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.n_samples = self.X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            indices = self.rng.permutation(self.X.shape[0])
            self.X = self.X.iloc[indices]
            self.y = self.y.iloc[indices]

        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start = self.current_batch * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            batch_X = Tensor(self.X[start:end].to_numpy(dtype=np.float32), requires_grad=False)
            batch_y = Tensor(self.y[start:end].to_numpy(dtype=np.int32).squeeze(), requires_grad=False)
            self.current_batch += 1
            return batch_X, batch_y
        else:
            raise StopIteration

    def __len__(self):
        return self.n_batches

def merge_datasets(datasets):
    merged_X = pd.concat([dataset.X for dataset in datasets], ignore_index=True)
    merged_y = pd.concat([dataset.y for dataset in datasets], ignore_index=True)
    return Dataset(merged_X, merged_y)

def load_breast_cancer_dataset():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    dataset = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets


    # Replace labels with integers
    label_mapping = {'B': 0, 'M': 1}
    y['Diagnosis'] = y['Diagnosis'].map(label_mapping)
    feature_names = X.columns.tolist()
    metadata = dataset.metadata
    metadata['feature_names'] = feature_names
    metadata['class_names'] = ['B', 'M']

    return Dataset(X, y), metadata

def normalize_features(dataset, mean, std):
    normalized_X = (dataset.X - mean) / (std + 1e-8)
    return Dataset(normalized_X, dataset.y)

def random_split(dataset, lengths, generator=None):
    
    n_samples = sum(lengths)  
    
    if generator is None:
        generator = np.random.default_rng()
    indices = generator.permutation(n_samples)
    X, y = dataset.X, dataset.y
    X = X.iloc[indices]
    y = y.iloc[indices]
    
    split_datasets = []
    for i in range(len(lengths)):
        start = sum(lengths[:i])
        end = start + lengths[i]
        split_datasets.append(Dataset(X.iloc[start:end], y.iloc[start:end]))
    return split_datasets

def permute_features(X, feature_names, random_state=None):
    rng = np.random.default_rng(random_state)
    for feature_name in feature_names:
        feature_index = X.columns.get_loc(feature_name)
        X.iloc[:, feature_index] = rng.permutation(X.iloc[:, feature_index].values)
    return X

def stratified_split(dataset, lengths, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    
    X, y = dataset.X, dataset.y
    unique_classes = y.unique()
    
    split_datasets = [Dataset([], []) for _ in range(len(lengths))]
    
    for cls in unique_classes:
        cls_indices = y[y == cls].index
        generator.shuffle(cls_indices)
        
        start = 0
        for i in range(len(lengths)):
            end = start + lengths[i] // len(unique_classes)
            split_datasets[i].X = pd.concat([split_datasets[i].X, X.loc[cls_indices[start:end]]])
            split_datasets[i].y = pd.concat([split_datasets[i].y, y.loc[cls_indices[start:end]]])
            start = end
            
    return split_datasets

def stratified_k_fold_iterator(dataset, k=5, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    
    lengths = [len(dataset.X) // k] * k 
    split_datasets = stratified_split(dataset, lengths, generator)

    for i in range(k):
        test_dataset = split_datasets[i]
        train_dataset = Dataset(pd.concat([split_datasets[j].X for j in range(k) if j != i]), 
                                pd.concat([split_datasets[j].y for j in range(k) if j != i]))
        yield train_dataset, test_dataset
