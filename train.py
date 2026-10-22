"""Command-line wrapper for `src.training.train`."""

from src.training.train import _config_from_args, train

if __name__ == "__main__":
    _, _, metrics = train(_config_from_args())
    print(metrics[-1])
