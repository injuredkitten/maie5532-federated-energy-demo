"""Measure an isolated public workload, optionally using the user's CUDA GPU."""
import argparse
import json
from .model import calibrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['numpy', 'torch-cpu', 'cuda'], default='numpy')
    parser.add_argument('--output')
    args = parser.parse_args()
    # Include the orchestration runtime in CPU reference RSS.
    import flwr  # noqa: F401
    result = calibrate(args.backend)
    if args.output:
        from .state import atomic_json
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
