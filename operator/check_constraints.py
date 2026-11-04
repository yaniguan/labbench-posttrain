"""Score a skill/waypoint plan against trusted workspace constraints."""
import argparse
import json
import sys

sys.path.insert(0, "src")
from labpt.constraints import check_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--constraints", required=True)
    args = parser.parse_args()
    print(json.dumps(check_constraints(json.load(open(args.plan)), json.load(open(args.constraints))), indent=2))


if __name__ == "__main__":
    main()
