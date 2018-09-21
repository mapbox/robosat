"""Log facilitator
"""

import os


class Log:

    """Create a log instance on a log file
    """

    def __init__(self, path, stdout=True):
        self.stdout = stdout
        self.fp = open(path, "a")
        assert self.fp, "Unable to open log file"

    """Log a new message to the opened log file, and optionnaly on stdout too
    """

    def log(self, msg):
        assert self.fp, "Unable to write in log file"
        self.fp.write(msg + os.linesep)
        self.fp.flush()

        if self.stdout:
            print(msg)
