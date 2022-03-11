# -*- coding: utf-8 -*-
"""logging_setup.py docstring
Module created in order to setup the logger used in the package.
"""


import logging
import logging.config


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.
    The individual arguments of `log_file`, `console`, `log_level` will
    overwrite the ones in cfg.
    Parameters
    ----------
        logger:
            Predefined logger object if present. If None a new logger
            object will be created from root.
        cfg: dict()
            Configuration of the logging to be implemented by default
        log_file: str
            Path to the log file for logs to be stored
        console: bool
            To include a console handler(logs printing in console)
        log_level: str
            One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
            default - `"DEBUG"`
    Returns
    -------
    logging.Logger
    """
    if not cfg:
        logging.config.fileConfig(
            "values.conf", disable_existing_loggers=False)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        logger.addHandler(fh)

    if not console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

    return logger