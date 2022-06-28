import inspect
import logging
import time
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "IDIOMATIC.%(name)-12s %(levelname)-8s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stream": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "": {"handlers": ["stream"], "level": "DEBUG", "propagate": False},
    },
}


def configure_logger_basic(format=DEFAULT_LOG_FORMAT, level=DEFAULT_LOG_LEVEL):
    logging.basicConfig(format=format, level=level)


def _get_info_about_calling_function():
    """
    extract the __name__ and __main__ from
    the calling function (for logging information)
    """
    call_stack = inspect.stack()
    # crawl up the call stack until the "log.py" module is found, then increment
    # the stack index one to get to the function that invoked this module
    # the file extension must be stripped in case the file is compiled to a .pyc
    current_file = str(Path(__file__).with_suffix(""))
    calling_frame_idx = (
        max(
            [
                idx
                for idx, frame in enumerate(call_stack)
                if str(Path(frame[1]).with_suffix("")) == current_file
            ]
        )
        + 1
    )
    calling_frame_globals = call_stack[calling_frame_idx][0].f_globals
    return calling_frame_globals["__name__"], calling_frame_globals["__file__"]


def get_logger(mod_name=None):
    if mod_name is None:
        mod_name, file_name = _get_info_about_calling_function()
        if mod_name == "__main__":
            mod_name = str(Path(file_name).resolve())
    logger = logging.getLogger(mod_name)
    logger.addHandler(logging.NullHandler())
    return logger


def configure_and_get_logger(
    mod_name=None, format=DEFAULT_LOG_FORMAT, level=DEFAULT_LOG_LEVEL
):
    logging.Formatter.converter = time.gmtime  # force timestamps to be UTC
    configure_logger_basic(format=format, level=level)
    return get_logger(mod_name)
