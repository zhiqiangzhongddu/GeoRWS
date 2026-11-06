import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zsl.run import run_zsl_from_cli

if __name__ == '__main__':
    raise SystemExit(run_zsl_from_cli(sys.argv[1:]))
