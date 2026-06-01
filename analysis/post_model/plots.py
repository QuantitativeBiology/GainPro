import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from filters import filter_artificial_missing_entries, filter_tissue_entries
from config import PlotConfig

TRAIN_COLOR = "#fc8b64"
VAL_COLOR = "#909cc5"
config = PlotConfig()

# =================================================
# =                 1. Loss                       =
# =================================================

def plot_loss(
    name_loss: str,
    train_loss: list,
    val_loss: list=None,
    figsize: tuple=(6,4),
) -> None:
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=figsize)
    plt.xlabel('Epoch')
    plt.ylabel(name_loss)
    plt.plot(epochs, train_loss, label="Train", color=TRAIN_COLOR)
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Validation", color=VAL_COLOR)
    plt.legend()
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

# =================================================
# =              2. Holdout                       =
# ================================================= 

def darken_color(color, factor=0.7) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

def build_longform_plot_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    # Expand "true_value" and "predicted_value" list into long format
    rows = []
    for _, row in df.iterrows():

        true_vals = row["true_value"]
        pred_vals = row["predicted_value"]

        for t, p in zip(true_vals, pred_vals):
            rows.append({
                "true_value": t,
                "predicted_value": p,
                "tissue": row["tissue"],
                "model": row["model"],
                "n_samples": row["n_samples"],
            })

    plot_df = pd.DataFrame(rows)
    return plot_df

def plot_scatter(
    df: pd.DataFrame,
    plot_dir: Path,
    figsize: tuple=(6,4),
    metrics: dict=None,
) -> None:
    model_name = df["model"].unique()[0]
    fig, ax = plt.subplots(figsize=figsize)

    x = df["true_value"].to_numpy(dtype=float)
    y = df["predicted_value"].to_numpy(dtype=float)
    pad = 0.15 # to prevent circles from getting cut
    lo = min(x.min(), y.min()) - pad
    hi = max(x.max(), y.max()) + pad

    ax.plot(
        [lo, hi], [lo, hi],
        color="red",
        linestyle="--",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.scatter(
        x=x,
        y=y,
        alpha=0.5, 
        s=18,
        color=config.model_color[model_name],
        edgecolor="none", 
    )
    sns.kdeplot(
        x=x,
        y=y,
        ax=ax,
        levels=5,
        color=darken_color(config.model_color[model_name]),
        linewidths=0.8,
        alpha=0.6,
        zorder=2,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(model_name)

    if metrics is not None:
        lines = [f"{k}: {v:.3f}" for k, v in metrics.items()]
        annotation = "\n".join(lines)
        ax.text(
            0.04,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#cccccc",
                linewidth=0.8,
                alpha=0.92,
            ),
        )
    
    dir = Path(f"{plot_dir}/{model_name.lower()}")
    dir.mkdir(parents=True, exist_ok=True)
    out_path = f"{dir}/observed_vs_predicted.png"
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

def plot_metric(
    df: pd.DataFrame, 
    metric: str, 
    miss_level: int, 
    plot_dir: Path,
    figsize: tuple=(4,4),
) -> None:
    """
    Bar chart of `metric` ("pearson_r" or "rmse") for a given missing level.

    Args:
        - df (DataFrame): full performance table (may span multiple miss levels)
        - metric (str): column to plot
        - miss_level (int): which miss level to filter / label in the filename
        - plot_dir (Path)
    """
    df = df[df["missing_level"] == miss_level].copy()
    df = df.sort_values(metric, ascending=(metric == "pearson_r"))

    plt.figure(figsize=figsize)
    ax = plt.gca()

    x = df[metric]
    y = np.arange(len(df))

    colors = [config.model_color[model] for model in df["model"]]

    ax.barh(y, x, color=colors, height=0.5)

    for _, (val, y_i) in enumerate(zip(x, y)):
        ax.annotate(
            f"{val:.3f}",
            (val, y_i),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(3, 0),
            textcoords="offset points",
        )
    
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_ylabel("")
    ax.set_xlabel(
        "Pearson’s r" if metric == "pearson_r"
        else "RMSE" if metric == "rmse"
        else metric.upper()
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    metric_slug = metric.lower().replace(" ", "_").replace("\u2019", "")
    path = plot_dir / f"{metric_slug}_by_model_miss{miss_level}.png"
    plt.savefig(path)
    print(f"Saved: {path}")

# =================================================
# =              2.1 tissue                       =
# =================================================

def _add_best_model_markers(
    ax,
    tissue_performance: pd.DataFrame,
    tissue_order,
    metric: str,
    best_fn
) -> None:
    """
    Annotate best-performing model bars with marker symbols.

    This helper inspects the performance values for each model across tissues,
    identifies the best model per tissue based on the provided selection function,
    and adds a marker above the corresponding bar in the bar plot.

    Args:
        - ax: Matplotlib Axes object containing the plotted bar containers.
        - tissue_performance (pd.DataFrame): DataFrame with columns ["tissue", "model", metric].
        - tissue_order: Ordered sequence of tissue labels matching the bar order in the plot.
        - metric (str): Metric column used to determine the best model per tissue.
        - best_fn: Function applied to the pivoted metric DataFrame that returns the best
            model for each tissue. Uses .idxmax() for metrics where higher
            is better and .idxmin() for metrics where lower is better.
    """
    pivot_df = tissue_performance.pivot(index="tissue", columns="model", values=metric)
    best_model_per_tissue = best_fn(pivot_df).to_dict()
    for container, model_name in zip(ax.containers, tissue_performance["model"].unique()):
        for bar, tissue in zip(container, tissue_order):
            if best_model_per_tissue.get(tissue) == model_name:
                marker_map = PlotConfig().marker_map
                marker = marker_map.get(model_name, "*")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    marker,
                    ha="center",
                    fontsize=9,
                )

def plot_metric_by_tissue(
    tissue_performance: pd.DataFrame,
    metric: str,
    miss_level: int,
    plot_dir: Path,
    # reference_model="tissue Mean", 
    yticks: list=None,
    figsize: tuple=(9,5)
) -> None:
    """
    Bar chart of `metric` ("pearson_r" or "rmse") per tissue for a given missing level.
    tissues are sorted by number of samples (ascending).
    The best model per tissue is annotated with its marker symbol.

    Args:
        - tissue_performance (pd.DataFrame):  full performance table (may span multiple miss levels).
        - metric (str): column to plot.
        - miss_level (int):  which miss level to filter / label in the filename.
        - plot_dir (Path): output directory.
        - reference_model (str): model used to determine tissue sort order.
        - yticks (list or None): custom y-tick positions
    """
    # Filter to the requested missing level if the column exists
    if "missing_level" in tissue_performance.columns:
        df = tissue_performance[tissue_performance["missing_level"] == miss_level].copy()
    else:
        df = tissue_performance.copy()

    order_df = (df[df["tissue"].unique()].sort_values("n_samples"))

    # order_df = (
    #     df[df["model"] == reference_model]
    #     .sort_values("n_samples")
    # )
    tissue_order = order_df["tissue"].tolist()
    x_labels = [
        f"{t} (N = {n})"
        for t, n in zip(order_df["tissue"], order_df["n_samples"])
    ]

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.barplot(data=df, x="tissue", y=metric, hue="model (Marker)",
                order=tissue_order, ax=ax)

    plt.xticks(range(len(x_labels)), x_labels, ha="center", fontsize=9)
    if yticks:
        plt.yticks(yticks)
    plt.ylabel("Pearson\u2019s r" if metric == "pearson_r" else metric)
    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.legend(title="model", loc="lower left")

    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    best_fn = (lambda p: p.idxmax(axis=1)) if metric == "pearson_r" else (lambda p: p.idxmin(axis=1))
    _add_best_model_markers(ax, df, tissue_order, metric, best_fn)

    plt.tight_layout()
    metric_slug = metric.lower().replace(" ", "_").replace("\u2019", "")
    path = plot_dir / f"{metric_slug}_by_tissue_miss{miss_level}.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")


def plot_scatter_by_tissue(
    df: pd.DataFrame,
    out_dir: Path,
    n_cols=3,
) -> None:
    plot_df = build_longform_plot_data(df)

    n_tissues = len(df["tissue"].unique())
    n_plots = n_tissues
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=False,
        sharey=False
    )
    fig.subplots_adjust(hspace=0.4)
    axes = axes.flatten()

    for i, tissue in enumerate(df["tissue"].unique()):
        ax = axes[i]
        tissue_df = plot_df[plot_df["tissue"] == tissue]
        num_samples = tissue_df["n_samples"].iloc[0]

        colors = [config.model_color[model] for model in df["model"]]

        sns.scatterplot(
            data=tissue_df,
            x="true_value",
            y="predicted_value",
            hue="model",
            alpha=0.5,
            s=25,
            edgecolor="none",
            ax=ax
        )
        for model_name, model_df in tissue_df.groupby("model"):
            sns.kdeplot(
                x=model_df["true_value"],
                y=model_df["predicted_value"],
                ax=ax,
                levels=4,
                color=colors[model_name],
            )

        true_values = tissue_df["true_value"]
        predicted_values = tissue_df["predicted_value"]
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())

        ax.plot([min_val, max_val], [min_val, max_val],
                color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{tissue}\n(N={num_samples})", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend_.remove()
    

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Observed", fontsize=10, y=0.0)
    fig.supylabel("Predicted", fontsize=10, x=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
    #             prop={"weight": "normal"})
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels))

    plt.tight_layout()
    plt.savefig(out_dir / f"observed_vs_predicted_tissue_scatter.png")

def plot_violin_by_tissue(
    model_dir: Path,
    plot_dir: Path, 
    n_cols=3
):
    """
    Observed vs. predicted distribution violin plots per tissue for a given ProtoGain missing level.
    """
    all_plots = _collect_tissue_plot_data(model_dir)

    n_plots = len(all_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=False,
        sharey=False
    )
    fig.subplots_adjust(hspace=0.4)
    axes = axes.flatten()

    for i, (tissue, num_samples, true_values, predicted_values) in enumerate(all_plots):
        ax = axes[i]
        violin_data = pd.DataFrame({
            "Value": np.concatenate([true_values, predicted_values]),
            "Type": (["Observed"] * len(true_values)) + (["Predicted"] * len(predicted_values)),
            "group": "all",
        })
        sns.violinplot(data=violin_data, x="group", y="Value", hue="Type",
                       split=True, inner="quartile", ax=ax,
                       palette={"Observed": "tab:blue", "Predicted": "tab:orange"})
        ax.set_xticks([])
        ax.set_title(f"{tissue}\n(N={num_samples})", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend_.remove()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               prop={"weight": "normal"})

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = plot_dir / f"predicted_vs_observed_distribution_protogain.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")

def _collect_tissue_plot_data(
    dir: Path, 
) -> list:
    """
    Return [(tissue, num_samples, true_values, predicted_values), ...] sorted
    descending by num_samples.
    """
    all_plots = []
    predictions = pd.read_csv(dir / "predictions" / "predictions.csv")
    for tissue in predictions["group_id"].unique():
        tissue_entries = filter_tissue_entries(predictions, tissue)
        num_samples = tissue_entries["sample_id"].nunique()
        true_values = tissue_entries["true_value"].to_numpy(dtype=float)
        predicted_values = tissue_entries["predicted_value"].to_numpy(dtype=float)
        all_plots.append((tissue, num_samples, true_values, predicted_values))

    return sorted(all_plots, key=lambda x: x[1], reverse=True)