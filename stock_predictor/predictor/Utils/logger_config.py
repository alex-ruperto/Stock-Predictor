# imports
import logging
from io import StringIO

# create a sharde log stream for the entire application
shared_log_stream = StringIO()

# set up a centralized logger configuration across the entire system
def configure_logger(name: str, log_stream: StringIO = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # set the logging level to include info

    # formatter setup
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # console logger setup
    if not logger.hasHandlers():
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
        # add handler for StringIO if provided
        if log_stream is not None:
            stream_handler = logging.StreamHandler(log_stream)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger

# initialize a shared logger once
configure_logger("Global", shared_log_stream)