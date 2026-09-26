import logging
import sys
import os


def setup_log(script_name):
    try:
        logger = logging.getLogger(script_name)

        if not logger.handlers:
            logger.setLevel(logging.DEBUG)

            log_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logs"
            )

            os.makedirs(log_folder, exist_ok=True)

            log_file = os.path.join(
                log_folder,
                f"{script_name}.log"
            )

            handler = logging.FileHandler(
                log_file,
                mode="w"
            )

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )

            handler.setFormatter(formatter)

            logger.addHandler(handler)

            logger.propagate = False

        return logger

    except Exception as e:
        error_type, error_message, error_traceback = sys.exc_info()
        raise e