import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import time

dynamic = False

parser = argparse.ArgumentParser(description="Plot loss curves from log file.")
parser.add_argument("--log", type=str, required=True, help="Path to the log file (CSV format)")
args = parser.parse_args()

log_path = args.log

if not os.path.exists(log_path):
    raise FileNotFoundError(f"log file does not exist: {log_path}")

try:
    while True:
        df = pd.read_csv(log_path)
        loss_columns = [col for col in df.columns if col.startswith("loss_") and col != "loss_total"]
        if loss_columns:
            df[loss_columns] = df[loss_columns].replace(0, pd.NA).ffill().fillna(0)

        plt.clf()
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(10, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [10, 5, 3]}
        )

        alpha = 1.0

        series_to_plot = [
            ("loss_sdf", "SDF Loss", "-"),
            ("loss_zero", "Zero Constraint", "-"),
            ("loss_eikonal", "Eikonal", "-"),
            ("loss_normal", "Normal", "-"),
            ("loss_sparse", "Sparse", "-"),
            ("loss_color_geo", "Color Geometry", "-"),
            ("loss_neg_sdf", "Negative SDF", "-"),
            ("loss_singular_hessian", "Singular Hessian", "-"),
            ("loss_rgb_jacobian", "RGB Jacobian", "-"),
            ("loss_rgb_gt", "RGB gt", "-"),
            ("loss_smooth", "Smooth", "--"),
        ]
        for column, label, linestyle in series_to_plot:
            if column in df.columns:
                ax1.plot(df["epoch"], df[column], label=label, alpha=alpha, linewidth=1, linestyle=linestyle)

        ax1.set_yscale("log")
        ax1.grid(True)
        ax1.legend()

        # ====

        ax2.plot(df["epoch"], df["loss_total"], label="Total Loss", color="black", linewidth=3)
        ax2.set_yscale("log")
        ax2.grid(True)
        ax2.legend()

        # ====
        ax3.plot(df["epoch"], df["learning_rate"], color='blue', label="Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.set_title("Learning Rate Schedule")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        
        if dynamic:
            plt.show(block=False)  
            plt.pause(0.1)         
            time.sleep(60)         
        else:
            plt.show()
            
except KeyboardInterrupt:
    print("\nEnd plotting.")
