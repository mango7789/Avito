"""Keep one TensorBoard endpoint following output/<model>/current."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("output/current"))
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    child, active = None, None
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"}
    try:
        while True:
            target = args.run.resolve() if args.run.exists() else None
            if target != active or child is None or child.poll() is not None:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                active = target
                if target is not None:
                    print(f"TensorBoard current run: {target}", flush=True)
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "tensorboard.main",
                            "--logdir",
                            str(target / "tensorboard"),
                            "--host",
                            "0.0.0.0",
                            "--port",
                            str(args.port),
                            "--reload_interval",
                            "10",
                            "--load_fast=false",
                        ],
                        env=environment,
                    )
            time.sleep(3)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)


if __name__ == "__main__":
    main()
