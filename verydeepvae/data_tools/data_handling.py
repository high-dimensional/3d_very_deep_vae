"""These are tools related to loading, formatting and interrogating data on disk."""

import os
import shutil


def delete_directory_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to empty directory %s. Reason: %s" % (file_path, e))
