import os
import numpy as np

from src.config import processed_data_dir


def load_dataset(env_name, dataset_name):
    # FIXME fname = os.path.join(processed_data_dir, ("%s_%s.npz" % (env_name, dataset_name)))
    fname = os.path.join(processed_data_dir, ("%s.npz" % dataset_name))
    print("Loading %s ..." % fname)
    dataset = np.load(fname)
    assert(dataset['angles'].shape[0] == dataset['images'].shape[0])
    print("Loaded %d datapoints from %s" % (dataset['angles'].shape[0], fname))
    return dataset


def load_datasets(env_name):
    train = load_dataset(env_name, 'linspaced_500')
    test = load_dataset(env_name, 'rand_1000')
    grid = load_dataset(env_name, 'grid_100_1000')

    return {'train': train, 'test': test, 'grid': grid}
