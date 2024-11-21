# -*- coding: utf-8 -*-
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_testing = logging.getLogger(__name__)
logger_testing.setLevel(logging.DEBUG)


def log_example():
    # insert these commands in between your functions. Mostly we wil use info but if there are some checks that we do
    # like if x=y or x>y and if it doesn't meet the criteria then we can add in logger.warning or error
    logger.debug("This message should go to the log file")
    logger.info("So should this")
    logger.warning("And this, too")
    logger.error("And non-ASCII stuff, too, like Øresund and Malmö")
