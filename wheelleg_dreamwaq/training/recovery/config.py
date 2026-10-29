from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import default_resolved_config, load_resolved_config, stable_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="创建或检查起身完整解析配置")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", metavar="PATH")
    group.add_argument("--check", metavar="PATH")
    args = parser.parse_args()
    if args.create:
        path = Path(args.create)
        if path.exists():
            raise FileExistsError(path)
        write_json(path, default_resolved_config())
        print(str(path.resolve()))
    else:
        print(stable_hash(load_resolved_config(args.check)))


if __name__ == "__main__":
    main()
