"""Run all Lab02 stages, or invoke each module separately with the same run ID."""
import argparse
from .preprocessing import prepare_run
from .tuning import tune_run
from .train import train_run
from .evaluate import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    run = prepare_run(args.config)
    tune_run(run)
    train_run(run)
    evaluate_run(run)
    print("Run:", run)


if __name__ == "__main__":
    main()
