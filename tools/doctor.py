"""Use after `pip install -e .`, or set PYTHONPATH=src."""
from food3d.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["doctor", *__import__("sys").argv[1:]]))

