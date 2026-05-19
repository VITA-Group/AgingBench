"""Allow `python -m agingbench …` as an alternative to the `agingbench`
console script.

This is the no-install entry point — useful for users who want to
smoke-test before committing to `pip install -e .`. Identical behavior
to the installed `agingbench` script.
"""
from .cli import main

if __name__ == "__main__":
    main()
