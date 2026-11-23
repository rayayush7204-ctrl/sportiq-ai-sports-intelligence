"""Script to extract frames from a video from command line."""

import sys
from sportiq.cli import main

if __name__ == "__main__":
    cli_args = ["extract-frames"] + sys.argv[1:]
    sys.exit(main(cli_args))
