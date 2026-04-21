import argparse
import re


def parse_metrics(path: str):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r"Global score:\s*J&F:\s*([0-9.]+)\s*J:\s*([0-9.]+)\s*F:\s*([0-9.]+)", text)
    if m is None:
        raise ValueError(f"Could not find Global score line in {path}")
    jf, j, f = map(float, m.groups())
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
    for k in ["J&F", "J", "F"]:
        delta = n[k] - b[k]
        print(f"{k:<8} {b[k]:>10.1f} {n[k]:>10.1f} {delta:>10.1f}")


if __name__ == "__main__":
    main()