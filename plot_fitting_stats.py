#!/usr/bin/env python3
"""
Plot optimization loss and fitted betas independently and save to transparent background PNG.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_optimizer_log(log_path):
    """
    Parse optimizer log file to extract loss values per iteration.
    """
    history = {"total": [], "data": [], "poZ_body": [], "betas": []}
    if not os.path.exists(log_path):
        return history

    pattern = re.compile(
        r"it (\d+) -- \[total loss = ([\d\.eE+-]+)\] - data = ([\d\.eE+-]+) \| .*? = ([\d\.eE+-]+) \| .*? = ([\d\.eE+-]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                history["total"].append(float(match.group(2)))
                history["data"].append(float(match.group(3)))
                # Extract named groups to handle varying order of betas/poZ_body
                if "betas =" in line and "poZ_body =" in line:
                    betas_match = re.search(r"betas = ([\d\.eE+-]+)", line)
                    poz_match = re.search(r"poZ_body = ([\d\.eE+-]+)", line)
                    if betas_match and poz_match:
                        history["betas"].append(float(betas_match.group(1)))
                        history["poZ_body"].append(float(poz_match.group(1)))
                else:
                    # Fallback to group indices if explicit search fails
                    history["poZ_body"].append(float(match.group(4)))
                    history["betas"].append(float(match.group(5)))
    return history


def plot_loss(loss_history, output_path, subject, trial):
    if not loss_history["total"]:
        print("Warning: No loss history to plot.")
        return

    plt.figure(figsize=(8, 4))
    iters = range(len(loss_history["total"]))
    plt.plot(iters, loss_history["total"], label="Total", color="black", linewidth=1.5)
    plt.plot(iters, loss_history["data"], label="Data", alpha=0.7)
    plt.plot(iters, loss_history["poZ_body"], label="Pose Prior", alpha=0.7)
    plt.plot(iters, loss_history["betas"], label="Beta Prior", alpha=0.7)

    plt.yscale("log")
    plt.ylim(1e-1, 1e3)
    plt.title(f"Optimizer Loss - Subject {subject} Trial {trial}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Display final loss values
    final_total = loss_history["total"][-1]
    final_data = loss_history["data"][-1]
    final_pose = loss_history["poZ_body"][-1]
    final_beta = loss_history["betas"][-1]

    loss_text = (
        f"Final Losses:\n"
        f"Total: {final_total:.2e}\n"
        f"Data:  {final_data:.2e}\n"
        f"Pose:  {final_pose:.2e}\n"
        f"Beta:  {final_beta:.2e}"
    )
    plt.gca().text(
        0.95,
        0.95,
        loss_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        family="monospace",
        bbox=dict(
            facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.3"
        ),
    )

    plt.savefig(output_path, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to {output_path}")


def plot_betas(betas, output_path, subject):
    if betas is None or len(betas) == 0:
        print("Warning: No betas to plot.")
        return

    plt.figure(figsize=(8, 4))
    beta_indices = range(len(betas))
    bars = plt.bar(beta_indices, betas, color="skyblue", edgecolor="navy")
    plt.title(f"Fitted Betas (Shape) - Subject {subject}")
    plt.xticks(beta_indices)
    plt.xlabel("Beta Index")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(
            bar.get_x() + bar.get_width() / 2.0,
            height if height > 0 else height - 0.2,
            f"{height:.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    plt.savefig(output_path, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved betas plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot optimization loss and betas independently."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Input directory (e.g., data/fitted_smpl_all_3)",
    )
    parser.add_argument(
        "-s", "--subject", type=str, required=True, help="Subject ID (e.g., 01)"
    )
    parser.add_argument(
        "-t", "--trial", type=str, required=True, help="Trial ID (e.g., 0)"
    )
    parser.add_argument("--loss_out", type=str, help="Output path for loss plot PNG")
    parser.add_argument("--beta_out", type=str, help="Output path for beta plot PNG")
    args = parser.parse_args()

    subj_id = f"SUBJ{args.subject}"
    subj_num = args.subject.lstrip("0") or "0"

    # Paths
    log_path = os.path.join(
        args.input_dir, subj_id, f"SUBJ{subj_num}_{args.trial}_optimizer.log"
    )
    betas_path = os.path.join(args.input_dir, subj_id, "betas.npy")
    params_path = os.path.join(
        args.input_dir, subj_id, f"SUBJ{subj_num}_{args.trial}_smpl_params.npz"
    )

    # Load and plot Loss
    if args.loss_out:
        loss_history = parse_optimizer_log(log_path)
        if loss_history["total"]:
            plot_loss(loss_history, args.loss_out, args.subject, args.trial)
        else:
            print(f"Error: Optimizer log not found or empty at {log_path}")

    # Load and plot Betas
    if args.beta_out:
        betas = None
        if os.path.exists(betas_path):
            betas = np.load(betas_path)
        elif os.path.exists(params_path):
            data = np.load(params_path)
            if "betas" in data:
                betas = data["betas"]

        if betas is not None:
            plot_betas(betas, args.beta_out, args.subject)
        else:
            print(f"Error: Betas not found for subject {args.subject}")


if __name__ == "__main__":
    main()
