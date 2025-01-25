import sys
from src.logger import logging
import traceback

def error_message_detail(error, error_detail):
    """
    Generate detailed error messages with file name and line number.
    """
    exc_tb = error_detail.__traceback__
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    return error_message






import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        """
        Custom Exception class to handle and log errors with traceback details.
        Args:
            error_message (str): Description of the error.
            error_detail (Exception): Original exception object.
        """
        super().__init__(error_message)
        if error_detail:
            self.error_message = f"{error_message}\nDetails:\n{traceback.format_exc()}"
        else:
            self.error_message = error_message

    def __str__(self):
        return self.error_message
