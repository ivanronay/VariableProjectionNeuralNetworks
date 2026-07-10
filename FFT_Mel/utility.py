import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import json
import time
from statistics import mean, stdev

def plot_FFT(loader, model, sample_rate=1/1):
    x, y = next(iter(loader))

    X = torch.fft.rfft(x[0])
    frequencies = torch.fft.rfftfreq(x.shape[1], d=sample_rate)

    plt.figure()
    plt.plot(frequencies, torch.abs(X))
    plt.xlabel("Frequency (Normalised?)")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude")
    plt.grid(True)
    plt.show()

    pred = model(x)
    pred = torch.round(pred)
    
    return pred

def log_fft_fold(
    filename,
    *,
    run_id,
    dataset_name,
    model_name,
    fold,
    n_folds,
    train_size,
    test_size,
    learning_rate,
    batch_size,
    epochs,
    hidden_layers,
    train_loss,
    train_accuracy,
    test_loss,
    test_accuracy,
    sensitivity_1,
    positive_predictivity_1,
    sensitivity_0,
    positive_predictivity_0,
):
    """
    Append one FFT K-fold result to a CSV file.
    """
    output_path = Path(filename)
    best_epoch_index = np.argmax(test_accuracy)

    row = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "fold": fold,
        "n_folds": n_folds,
        "train_size": train_size,
        "test_size": test_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "hidden_layers": " ".join(hidden_layers),

        # Best epoch information
        "best_epoch_index": best_epoch_index,
        "best_train_loss": train_loss[best_epoch_index],
        "best_test_loss": test_loss[best_epoch_index],
        "best_train_accuracy": train_accuracy[best_epoch_index],
        "best_test_accuracy": test_accuracy[best_epoch_index],

        # Class 1 metrics at the selected epoch
        "class_1_sensitivity": sensitivity_1[best_epoch_index],
        "class_1_positive_predictivity": positive_predictivity_1[best_epoch_index],

        # Class 0 metrics at the selected epoch
        "class_0_sensitivity": sensitivity_0[best_epoch_index],
        "class_0_positive_predictivity": positive_predictivity_0[best_epoch_index],

        # Full histories
        "train_loss_history": " ".join(train_loss),
        "test_loss_history": " ".join(test_loss),
        "train_accuracy_history": " ".join(train_accuracy),
        "test_accuracy_history": " ".join(test_accuracy),
    }

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return row

def log_fft_summary(
    filename,
    *,
    run_id,
    dataset_name,
    model_name,
    fold_rows,
):
    """Write one summary row for a complete FFT cross-validation run."""

    output_path = Path(filename)
    
    accuracies = [
        float(row["best_test_accuracy"])
        for row in fold_rows
    ]

    class_1_sensitivity = [
        float(row["class_1_sensitivity"])
        for row in fold_rows
    ]

    class_1_ppv = [
        float(row["class_1_positive_predictivity"])
        for row in fold_rows
    ]

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "n_folds": len(fold_rows),
        "mean_test_accuracy": mean(accuracies),
        "std_test_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_test_accuracy": min(accuracies),
        "max_test_accuracy": max(accuracies),
        "mean_class_1_sensitivity": mean(class_1_sensitivity),
        "mean_class_1_positive_predictivity": mean(class_1_ppv),
        "fold_accuracies": json.dumps(accuracies),
    }

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return row