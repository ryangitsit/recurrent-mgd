import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datasets import load_dataset

import src.utils.helper_functions as hf


def one_hot_y(y_targets):
    """
    Converts labels to one-hot encoded format.

    Args:
        y_targets (numpy.ndarray): The labels to convert.

    Returns:
        numpy.ndarray: The one-hot encoded labels.
    """
    num_classes = len(list(set(y_targets)))
    hot = jnp.eye(num_classes)[y_targets]
    return hot

def shuffle(data,labels):
    shuff = np.arange(len(labels))
    np.random.shuffle(shuff)
    data   = data[shuff]
    labels = labels[shuff]
    return data, labels

def split_data(X,y_targets):
    split = int(0.8*len(X))
    trainX = X[:split]
    trainy = y_targets[:split]
    testX  = X[split:]
    testy  = y_targets[split:]
    return trainX, trainy, testX, testy

@jax.jit
def normalize_batch(x,margin=1e-8):
    mean = jnp.mean(x, axis=1, keepdims=True)
    std  = jnp.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + margin)

def get_ECG5000():
    """
    Loads and preprocesses the ECG5000 dataset.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The ECG data.
            - labels (numpy.ndarray): The corresponding labels.
    """

    dataframe = pd.read_csv(
        'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', 
        header=None
    )

    raw_data = dataframe.values
    dataframe.head()
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]

    shuff = np.arange(len(labels))
    np.random.shuffle(shuff)
    data   = data[shuff]
    labels = labels[shuff]

    return data, labels

def get_ECG5000_multi():
    """
    Loads and preprocesses the ECG5000 dataset.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The ECG data.
            - labels (numpy.ndarray): The corresponding labels.
    """
    import pandas as pd
    import arff

    folder = "../../ECG5000"
    
    # Load TRAIN
    with open(f"{folder}/ECG5000_TRAIN.arff", 'r') as f:
        train_data = arff.load(f)

    # Load TEST
    with open(f"{folder}/ECG5000_TEST.arff", 'r') as f:
        test_data = arff.load(f)

    # print(test_data,train_data)
    # Convert to pandas
    df_train = pd.DataFrame(train_data["data"])
    df_test  = pd.DataFrame(test_data["data"])

    def df_to_data(df):
        raw_data = df.values
        labels = np.array(raw_data[:, -1], dtype=int)-1
        data   = np.array(raw_data[:, 0:-1], dtype=float)
        return data, labels
    
    x_train,y_train = df_to_data(df_train)
    x_test,y_test   = df_to_data(df_test)

    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test   = shuffle(x_test,y_test)

    y_train_hot = one_hot_y(y_train)
    y_test_hot  = one_hot_y(y_test)

    return x_train,y_train_hot,x_test,y_test_hot


def get_data(dataset,norm=True,save=False):
    """
    Loads and preprocesses a dataset.
    """
    try:
        with open(f"../../datasets/{dataset}_norm_{norm}.pkl", "rb") as f:
            data = pickle.load(f)
    except:
        if dataset == "ECG":
            U, y_targets = get_ECG5000()
            U_train, y0_train, U_test, y0_test = split_data(U,y_targets)
        
        elif dataset=="ECG_multi":
            U_train, y0_train, U_test, y0_test = get_ECG5000_multi()

        if norm==True:
            U_train = normalize_batch(U_train)
            U_test  = normalize_batch(U_test)

        data = dict(
            train_size=len(y0_train), test_size=len(y0_test),
            U_train=U_train, y0_train=y0_train, U_test=U_test, y0_test=y0_test)
        
        if save==True:
            hf.create_directory(["../../datasets"])
            with open(f"../../datasets/{dataset}_norm_{norm}.pkl", "wb") as f:
                pickle.dump(data, f)

    return data

class InfiniteLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = np.array(X)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.ptr = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > len(self.X):
            self.ptr = 0
            if self.shuffle:
                np.random.shuffle(self.indices)

        idx = self.indices[self.ptr:self.ptr+self.batch_size]
        self.ptr += self.batch_size

        xb = jnp.array(self.X[idx])
        yb = jnp.array(self.y[idx])
        return xb, yb

def load_mnist_in_memory(batch_size):
    cache_dir="~/.cache/huggingface/datasets"

    trainset = load_dataset("mnist", split = 'train', cache_dir = cache_dir)
    testset  = load_dataset("mnist", split = 'test', cache_dir = cache_dir)

    def preprocess(sample):
        sample['label'] = jax.nn.one_hot(sample['label'], 10)
        return sample

    trainset = trainset.map(preprocess).shuffle(seed=0)
    testset = testset.map(preprocess).shuffle(seed=0)

    # Make datasets output in numpy format
    trainset = trainset.with_format('numpy')
    testset  = testset.with_format('numpy')

    # Copy to memory
    trainset = trainset[:]
    testset  = testset[:]

    trainset['image'] = np.array(trainset['image'])/255
    testset['image']  = np.array(testset['image'])/255

    trainloader = InfiniteLoader(X=trainset['image'], y=trainset['label'], batch_size=batch_size, shuffle=True)
    testloader  = InfiniteLoader(X=testset['image'], y=testset['label'], batch_size=batch_size, shuffle=True)

    return trainloader, testloader



