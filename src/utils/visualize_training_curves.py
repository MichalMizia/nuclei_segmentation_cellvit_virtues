import numpy as np
import matplotlib.pyplot as plt


def plot_training_summary(by_experiment):
    """
    Plots:
    - One row per experiment:
        Left: Train & Val Loss
        Right: Validation Dice
    - Final row:
        Left: Mean Validation Loss (comparison)
        Right: Mean Validation Dice (comparison)

    Parameters
    ----------
    by_experiment : dict
        {
            experiment_name: [
                {
                    "train_losses": np.ndarray,
                    "val_losses": np.ndarray,
                    "val_dices": np.ndarray,
                },
                ...
            ],
            ...
        }
    """

    experiments = list(by_experiment.keys())
    n_exp = len(experiments)

    fig, axes = plt.subplots(
        nrows=n_exp + 1,
        ncols=2,
        figsize=(12, 3 * (n_exp + 1)),
        sharex=False
    )

    if n_exp == 1:
        axes = np.array([axes])

    # ==============================
    # PER-EXPERIMENT ROWS
    # ==============================

    for i, exp_name in enumerate(experiments):
        runs = by_experiment[exp_name]

        ax_loss = axes[i, 0]
        ax_dice = axes[i, 1]

        # ---- LOSS PLOTS ----
        for r in runs:
            epochs = np.arange(1, len(r["train_losses"]) + 1)
            ax_loss.plot(epochs, r["train_losses"], color="tab:blue", alpha=0.4)
            ax_loss.plot(epochs, r["val_losses"],   color="tab:orange", alpha=0.4)

        ax_loss.set_title(f"{exp_name} — Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)

        if i == 0:
            ax_loss.plot([], [], color="tab:blue", label="Train Loss")
            ax_loss.plot([], [], color="tab:orange", label="Val Loss")
            ax_loss.legend()

        # ---- DICE PLOTS ----
        for r in runs:
            epochs = np.arange(1, len(r["val_dices"]) + 1)
            ax_dice.plot(epochs, r["val_dices"], color="tab:green", alpha=0.6)

        ax_dice.set_title(f"{exp_name} — Validation Dice")
        ax_dice.set_xlabel("Epoch")
        ax_dice.set_ylabel("Dice")
        ax_dice.grid(True)

    # ==============================
    # FINAL COMPARISON ROW
    # ==============================

    ax_loss_cmp = axes[-1, 0]
    ax_dice_cmp = axes[-1, 1]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, exp_name in enumerate(experiments):
        runs = by_experiment[exp_name]
        color = color_cycle[idx % len(color_cycle)]

        # ---- VALIDATION LOSS (mean) ----
        max_len = max(len(r["val_losses"]) for r in runs)
        loss_matrix = []

        for r in runs:
            l = r["val_losses"]
            padded = np.full(max_len, np.nan)
            padded[:len(l)] = l
            loss_matrix.append(padded)

        loss_matrix = np.array(loss_matrix)
        mean_loss = np.nanmean(loss_matrix, axis=0)
        epochs = np.arange(1, len(mean_loss) + 1)

        ax_loss_cmp.plot(
            epochs,
            mean_loss,
            color=color,
            linewidth=1.5,
            label=exp_name
        )

        # ---- VALIDATION DICE (mean) ----
        max_len = max(len(r["val_dices"]) for r in runs)
        dice_matrix = []

        for r in runs:
            d = r["val_dices"]
            padded = np.full(max_len, np.nan)
            padded[:len(d)] = d
            dice_matrix.append(padded)

        dice_matrix = np.array(dice_matrix)
        mean_dice = np.nanmean(dice_matrix, axis=0)
        epochs = np.arange(1, len(mean_dice) + 1)

        ax_dice_cmp.plot(
            epochs,
            mean_dice,
            color=color,
            linewidth=1.5,
            label=exp_name
        )

    ax_loss_cmp.set_title("Validation Loss — Comparison")
    ax_loss_cmp.set_xlabel("Epoch")
    ax_loss_cmp.set_ylabel("Val Loss")
    ax_loss_cmp.grid(True)

    ax_dice_cmp.set_title("Validation Dice — Comparison")
    ax_dice_cmp.set_xlabel("Epoch")
    ax_dice_cmp.set_ylabel("Dice")
    ax_dice_cmp.grid(True)
    ax_dice_cmp.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
