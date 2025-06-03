import logging
import os
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def setup_logging(log_file_name_prefix="integration_test", level=logging.INFO):
    """
    Sets up logging to console and a file.

    Args:
        log_file_name_prefix (str): Prefix for the log file name. 
                                     Timestamp will be appended.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("PNET_Integration_Test")
    logger.setLevel(level)

    # Prevent multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{log_file_name_prefix}_{timestamp}.log")
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

if __name__ == '__main__':
    # Example usage:
    logger = setup_logging(level=logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print(f"Example log file created in: {LOGS_DIR}")
