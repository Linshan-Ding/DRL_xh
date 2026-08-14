from __future__ import annotations

import argparse

from agent.baselines.dqn import build_dqn_parser, train_dqn


def build_parser() -> argparse.ArgumentParser:
    return build_dqn_parser("ddqn")


def train(args) -> None:
    train_dqn(args, algorithm="ddqn", double_dqn=True)


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
