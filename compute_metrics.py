import csv
from collections import defaultdict

INPUT_FILE = "responses_labelled.csv"
COND_COL = "condition"
LABEL_COL = "label"  # C, H, B


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows):
    groups = defaultdict(list)
    for r in rows:
        cond = r[COND_COL]
        groups[cond].append(r)

    results = {}

    for cond, items in groups.items():
        total = len(items)
        num_C = sum(1 for r in items if r[LABEL_COL] == "C")
        num_H = sum(1 for r in items if r[LABEL_COL] == "H")
        num_B = sum(1 for r in items if r[LABEL_COL] == "B")
        num_H_or_B = num_H + num_B

        if total == 0:
            continue

        acc = num_C / total
        hall = num_H_or_B / total
        blr = num_B / total
        blr_cond = (num_B / num_H_or_B) if num_H_or_B > 0 else 0.0

        results[cond] = {
            "total": total,
            "C": num_C,
            "H": num_H,
            "B": num_B,
            "Acc": acc,
            "Hall": hall,
            "BLR": blr,
            "BLR_cond": blr_cond,
        }

    return results


def print_table(results):
    print("Condition        | Total | Acc  | Hall | BLR  | BLR_cond")
    print("---------------- | ----- | ---- | ---- | ---- | --------")
    for cond, stats in results.items():
        print(
            f"{cond:16s} | "
            f"{stats['total']:5d} | "
            f"{stats['Acc']:.2f} | "
            f"{stats['Hall']:.2f} | "
            f"{stats['BLR']:.2f} | "
            f"{stats['BLR_cond']:.2f}"
        )


def main():
    rows = load_rows(INPUT_FILE)
    results = compute_metrics(rows)
    print_table(results)


if __name__ == "__main__":
    main()
