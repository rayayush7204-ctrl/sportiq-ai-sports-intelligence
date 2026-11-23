"""Script to inspect video metadata from command line."""

import sys
from sportiq.cli import main

if __name__ == "__main__":
    cli_args = ["inspect-video"] + sys.argv[1:]
    sys.exit(main(cli_args))
