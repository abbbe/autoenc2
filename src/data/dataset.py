import os
import logging

import numpy as np

from src.config import processed_data_dir

logger = logging.getLogger(__name__)


def get_dataset_fname(env_name, dataset_name):
    return os.path.join(processed_data_dir, ("%s_%s.npz" % (env_name, dataset_name)))


def save_dataset(env_name, dataset_name, angles, images):
    fname = get_dataset_fname(env_name, dataset_name)
    assert(not os.path.exists(fname))  # FIXME
    logger.info("%d datapoints generated, saving in %s" %
                (angles.shape[0], fname))
    np.savez(fname, angles=angles, images=images)


def load_dataset(env_name, dataset_name):
    fname = get_dataset_fname(env_name, dataset_name)
    logger.info("Loading %s ..." % fname)
    dataset = np.load(fname)
    assert(dataset['angles'].shape[0] == dataset['images'].shape[0])
    logger.info("Loaded %d datapoints from %s" %
                (dataset['angles'].shape[0], fname))
    return dataset


def load_datasets(env_name):
    train = load_dataset(env_name, 'linspaced_500')
    test = load_dataset(env_name, 'rand_1000')
    grid = load_dataset(env_name, 'grid_100_1000')

    return {'train': train, 'test': test, 'grid': grid}
