"""Log facilitator
"""

import os
import sys


class Log:

    """Create a log instance on a log file
    """

    def __init__(self, path, out=sys.stdout, mode="a"):
        self.out = out
        self.fp = open(path, mode)
        assert self.fp, "Unable to open log file"

    """Log a new message to the opened log file, and optionnaly on stdout or stderr too
    """

    def log(self, msg):
        assert self.fp, "Unable to write in log file"
        self.fp.write(msg + os.linesep)
        self.fp.flush()

        if self.out:
            print(msg, file=self.out)
