"""Log facilitator
"""

import os
import sys


class Log:

    """Create a log instance on a log file
    """

    def __init__(self, path, out=sys.stdout):

        self.fp = None
        self.out = out
        try:
            if path:
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                self.fp = open(path, mode="a")
        except:
            sys.exit("Unable to write in log directory")

    """Log a new message to the opened log file, and optionnaly on stdout or stderr too
    """

    def log(self, msg):
        try:
            if self.fp:
                self.fp.write(msg + os.linesep)
                self.fp.flush()

            if self.out:
                print(msg, file=self.out)
        except:
            sys.exit("Unable to write in log file")
