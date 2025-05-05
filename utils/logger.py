import logging

def get_logger(log_file=None, log_level=logging.INFO, rank=0):
    logger = logging.getLogger(name="ViT")
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level if rank == 0 else logging.ERROR)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger