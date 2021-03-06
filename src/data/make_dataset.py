# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import numpy as np

from src.config import processed_data_dir, project_dir
from src.data.dataset import save_dataset, get_dataset_fname
from src.envs import get_env_names, load_env

logger = logging.getLogger(__name__)


def generate_linspaced_angles(params):
    N12 = params['N']
    # FIXME number of joints and low/high angles are hardcoded
    a12s = np.linspace(-1., 1., N12)

    N = N12 ** 2
    angles = np.zeros((N, 2))

    i = 0
    for i1 in range(N12):
        for i2 in range(N12):
            angles[i] = [a12s[i1], a12s[i2]]
            i += 1
    return angles


def generate_grid_angles(params):
    N1 = params['N1']
    N2 = params['N2']
    # FIXME number of joints and low/high angles are hardcoded
    a1s = np.linspace(-1., 1., N1)
    a2s = np.linspace(-1., 1., N2)

    N = N1 * N2
    angles = np.zeros((2*N, 2))

    i = 0
    for i1 in range(N1):
        for i2 in range(N2):
            angles[i] = [a1s[i1], a2s[i2]]
            angles[i + 1] = [a2s[i2], a1s[i1]]
            i += 2

    return angles


def generate_rand_angles(params):
    N = params['N']
    # FIXME number of joints is hardcoded
    return np.random.uniform(low=-1.0, high=1.0, size=(N, 2))


def generate_images(env, angles):
    N = angles.shape[0]
    # FIXME number of channels is hardcoded
    images = np.zeros((N, env.D, env.D, 1))

    for i in range(N):
        env.step(angles[i])
        env.render()
        image = env.get_image()

        images[i] = image

    return images


def generate_and_save(gen_method, params, env, env_name, name_template):
    """
    Generates angles using given 'gen_method' with given 'params'
    Uses given 'env' to produce images corresponding to these angles
    Saves angles and images in npz file name formatted with the same 'params'
    """
    dataset_name = name_template.format(**params)
    fname = get_dataset_fname(env_name, dataset_name)
    if os.path.exists(fname):
        logger.error("File %s exists, skipping dataset" % fname)
        return

    angles = gen_method(params)
    n_datapoints = angles.shape[0]
    logger.info("Generating '%d' datapoints for '%s' ..." %
                (n_datapoints, dataset_name))

    images = generate_images(env, angles)

    save_dataset(fname, angles, images)


@click.command()
# TODO: turn this tool into per-environment dataset configurator/generator
@click.option('--linspaced_steps', default=100, type=click.IntRange(min=1))
@click.option('--rand_count', default=15000, type=click.IntRange(min=1))
@click.option('--grid_steps1', default=20, type=click.IntRange(min=1))
@click.option('--grid_steps2', default=500, type=click.IntRange(min=1))
@click.argument('env_name', type=click.Choice(get_env_names()), required=True)
def main(env_name, linspaced_steps, rand_count, grid_steps1, grid_steps2):
    """ Generates angles/images datasets for given env
    """
    env = load_env(env_name)

    generate_and_save(generate_linspaced_angles, {
                      'N': linspaced_steps},
                      env, env_name, 'linspaced_{N}')
    generate_and_save(generate_grid_angles, {
                      'N1': grid_steps1, 'N2': grid_steps2},
                      env, env_name, 'grid_{N1}_{N2}')
    generate_and_save(generate_rand_angles, {
                      'N': rand_count}, env, env_name, 'rand_{N}')


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
