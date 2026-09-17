"""Avvio rapido: python run.py [2d|1d|demo|all]."""
import sys
from src.__main__ import main

if __name__ == "__main__":
    # default: tutto
    main(sys.argv[1:] or ["all"])
