import os
import logging


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(20)
logger.propagate = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
