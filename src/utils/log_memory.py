import logging
import subprocess

import psutil
import torch

logger = logging.getLogger(__name__)


def log_gpu_memory():

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_bytes = subprocess.check_output(
                f"nvidia-smi -i {i} --query-gpu=memory.free --format=csv,nounits,noheader",
                shell=True,
            )

            memory_float_gb = float(memory_bytes) / 1024
            logger.info(f"GPU {i} memory available: {memory_float_gb:.2f} GB")
    else:
        logger.info("No GPUs available")


def log_ram():
    memory_info = psutil.virtual_memory()

    logger.info(
        f"RAM available: {memory_info.available / (1024.0 ** 3):.2f} GB"
    )