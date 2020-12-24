import os, time
import logging, sys


def get_logger(logdir):
    # 1.create log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # 2. create log file
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # 3. create log
    logger = logging.getLogger('trian')
    logger.setLevel(logging.INFO)

    # 4. set log format
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 5. StreamHandler Output the log to console
    setream_handler = logging.StreamHandler(sys.stdout)
    setream_handler.setFormatter(formatter)
    logger.addHandler(setream_handler)

    # 6. FileHandler: Output the log to log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
