import fire

import logging

from ..utils.log_memory import log_gpu_memory, log_ram

# def train(data: str, output: str):
def train():
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("#" * 64)

    log_gpu_memory()
    log_ram()

    logger.info("#" * 64)


if __name__ == "__main__":
    fire.Fire(train)
