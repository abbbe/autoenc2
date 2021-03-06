# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.envs import get_env_names, load_env

#    logger = logging.getLogger(__name__)
#    logger.info('making final data set from raw data')


@click.command()
@click.argument('env_name', type=click.Choice(get_env_names()), required=True)
def main(env_name):
    env = load_env(env_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
