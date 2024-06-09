# imports
import logging

# set up a centralized logger configuration across the entire system
def configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # set the logging level to include info

    # console handler setup
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # formatter setup
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # add formatter to the console handler and add the console handler to the logger
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger
