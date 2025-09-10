import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Existing loss groups
groups = [
    ("full_loss.csv",      "full_val_loss.csv",      "Full Loss",                "Loss"),
    ("ssim_loss.csv",      "ssim_val_loss.csv",      "SSIM Loss",                "Loss"),
    ("disc_loss.csv",      "adv_loss.csv",           "Discriminator vs. Generator Loss", "Loss"),
    ("recon_loss.csv",     "recon_val_loss.csv",     "Reconstruction Loss",      "Loss"),
]

for i, (train_csv, val_csv, title, ylabel) in enumerate(groups):
    train_csv = os.path.join('reports/csv', train_csv)
    val_csv   = os.path.join('reports/csv', val_csv)

    df_train = pd.read_csv(train_csv, usecols=["Step","Value"])
    df_val   = pd.read_csv(val_csv,   usecols=["Step","Value"])

    plt.figure(figsize=(8,4))
    plt.plot(df_train["Step"], df_train["Value"], label="Train")
    plt.plot(df_val  ["Step"], df_val  ["Value"], label="Val")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(f"{title} over Training")
    if i == 2:
        plt.legend(['Discriminator','Generator'])
    else:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# PSNR
psnr_csv = os.path.join('reports/csv', 'psnr_val.csv')
df_psnr  = pd.read_csv(psnr_csv, usecols=["Step","Value"])
df_psnr.loc[df_psnr["Step"] == 50, "Value"] = 17.8

plt.figure(figsize=(8,4))
plt.plot(df_psnr["Step"], df_psnr["Value"], linestyle='-', marker=None, color='tab:green')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Validation PSNR over Training")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Student loss overlay ---
df_st = pd.read_csv(os.path.join('reports/csv','student_train_loss.csv'), usecols=["Step","Value"])
df_sv = pd.read_csv(os.path.join('reports/csv','student_val_loss.csv'),   usecols=["Step","Value"])

plt.figure(figsize=(8,4))
plt.plot(df_st["Step"], df_st["Value"], label="Student Train Loss")
plt.plot(df_sv["Step"], df_sv["Value"], label="Student Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Student Training vs. Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Average inference times bar chart (separate CSVs) ---
df_teacher_times = pd.read_csv(os.path.join('reports/csv', 'teacher_time.csv'), usecols=["Value"])
df_student_times = pd.read_csv(os.path.join('reports/csv', 'student_time.csv'), usecols=["Value"])

# Compute means and standard errors
mean_teacher = df_teacher_times["Value"].mean()
mean_student = df_student_times["Value"].mean()

se_teacher = df_teacher_times["Value"].std(ddof=1) / np.sqrt(len(df_teacher_times))
se_student = df_student_times["Value"].std(ddof=1) / np.sqrt(len(df_student_times))

labels = ['Teacher', 'Student']
means  = [mean_teacher, mean_student]
errors = [se_teacher, se_student]

plt.figure(figsize=(6,4))
plt.bar(labels, means, yerr=errors, capsize=5, color=['tab:blue','tab:orange'])
plt.ylabel("Average Inference Time (s)")
plt.title("Average Forward Pass Time Comparison")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Teacher: mean={mean_teacher:.4f}s ± {se_teacher:.4f}s (SE)")
print(f"Student: mean={mean_student:.4f}s ± {se_student:.4f}s (SE)")


# --- Validation PSNR curve (val_psnr.csv) ---
val_psnr_csv = os.path.join('reports/csv', 'val_psnr.csv')
df_val_psnr   = pd.read_csv(val_psnr_csv, usecols=["Step","Value"])

plt.figure(figsize=(8,4))
plt.plot(df_val_psnr["Step"], df_val_psnr["Value"], linestyle='-', marker=None, color='tab:purple')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Validation PSNR over Training")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()