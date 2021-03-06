import os
import logging

import numpy as np

from src.config import processed_data_dir

logger = logging.getLogger(__name__)


def get_dataset_fname(env_name, dataset_name):
    return os.path.join(processed_data_dir, ("%s_%s.npz" % (env_name, dataset_name)))


def save_dataset(fname, angles, images):
    logger.info("%d datapoints generated, saving in %s" %
                (angles.shape[0], fname))
    np.savez(fname, angles=angles, images=images)


def load_dataset(fname):
    logger.info("Loading from %s ..." % fname)
    dataset = np.load(fname)
    dataset = dict(dataset)  # defeat lazy loading
    assert(dataset['angles'].shape[0] == dataset['images'].shape[0])
    logger.info("Loaded %d datapoints from %s" %
                (dataset['angles'].shape[0], fname))
    return dataset


def load_datasets(env_name):
    #train = load_dataset(env_name, 'linspaced_500')
    #train = load_dataset(env_name, 'linspaced_100')

    train = load_dataset(get_dataset_fname(env_name, 'rand_15000'))
    test = load_dataset(get_dataset_fname(env_name, 'rand_1000'))
    grid = load_dataset(get_dataset_fname(env_name, 'grid_20_500'))

    return {'train': train, 'test': test, 'grid': grid}
