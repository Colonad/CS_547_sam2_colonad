import argparse
import re
from pathlib import Path


def read_text_auto(path: str) -> str:
    raw = Path(path).read_bytes()

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def parse_metrics(path: str):
    text = read_text_auto(path)

    pattern = r"Global score:\s*J&F:\s*([0-9.]+)\s*J:\s*([0-9.]+)\s*F:\s*([0-9.]+)"
    match = re.search(pattern, text)

    if match is None:
        raise ValueError(f"Could not find Global score line in {path}")

    jf, j, f = map(float, match.groups())
    return {"J&F": jf, "J": j, "F": f}


def main():
    parser = argparse.ArgumentParser(description="Compare Exp1 and Exp2 metrics.")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--new", type=str, required=True)
    args = parser.parse_args()

    b = parse_metrics(args.baseline)
    n = parse_metrics(args.new)

    print("Metric comparison")
    print("-----------------")
    print(f"{'Metric':<8} {'Baseline':>10} {'New':>10} {'Delta':>10}")
    print(f"{'J&F':<8} {b['J&F']:>10.1f} {n['J&F']:>10.1f} {n['J&F'] - b['J&F']:>10.1f}")
    print(f"{'J':<8}   {b['J']:>10.1f} {n['J']:>10.1f} {n['J'] - b['J']:>10.1f}")
    print(f"{'F':<8}   {b['F']:>10.1f} {n['F']:>10.1f} {n['F'] - b['F']:>10.1f}")


if __name__ == "__main__":
    main()