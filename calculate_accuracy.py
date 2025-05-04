import json
from collections import defaultdict

def calculate_accuracy(json_path="animal_output/results.json"):
    with open(json_path, "r") as f:
        results = json.load(f)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0

    print(f"\n📊 Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")

    # Class-wise accuracy
    class_totals = defaultdict(int)
    class_corrects = defaultdict(int)

    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted_class"]
        class_totals[gt] += 1
        if gt == pred:
            class_corrects[gt] += 1

    print("🔍 Class-wise Accuracy:")
    for cls in class_totals:
        acc = class_corrects[cls] / class_totals[cls] if class_totals[cls] else 0
        print(f" - {cls}: {acc * 100:.2f}% ({class_corrects[cls]}/{class_totals[cls]})")

if __name__ == "__main__":
    calculate_accuracy()
