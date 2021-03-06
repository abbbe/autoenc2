import os
from pathlib import Path

import logging

DATA_DIR = "data"
PROCESSED_DATA_DIR = "processed"
OUTPUT_DATA_DIR = "output"

project_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(project_dir, DATA_DIR)
processed_data_dir = os.path.join(data_dir, PROCESSED_DATA_DIR)
output_data_dir = os.path.join(data_dir, OUTPUT_DATA_DIR)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
