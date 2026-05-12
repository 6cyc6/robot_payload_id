import os


def safe_mkdir(dir_path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
