import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from filters import filter_artificial_missing_entries, filter_tissue_entries
from config import PlotConfig

FIGSIZE = (6,4)
TRAIN_COLOR = "#fc8b64"
VAL_COLOR = "#909cc5"

# =================================================
# =                 1. Loss                       =
# =================================================

def plot_loss(
    name_loss: str,
    train_loss: list,
    val_loss: list=None,
) -> None:
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=FIGSIZE)
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
    # Expand "True values" and "Predicted values" list into long format
    rows = []
    for _, row in df.iterrows():

        true_vals = row["True values"]
        pred_vals = row["Predicted values"]

        for t, p in zip(true_vals, pred_vals):
            rows.append({
                "True values": t,
                "Predicted values": p,
                "Tissue": row["Tissue"],
                "Model": row["Model"],
                "Number of samples": row["Number of samples"],
            })

    plot_df = pd.DataFrame(rows)
    return plot_df

def plot_scatter(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        data=df,
        x="true_value", 
        y="predicted_value",
        alpha=0.5, 
        s=25, 
        edgecolor="none", 
    )
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(f"{out_dir}/observed_vs_predicted.png")

def plot_metric(
    df: pd.DataFrame, 
    metric: str, 
    miss_level: int, 
    plot_dir: Path,
    yticks: list=None
) -> None:
    """
    Bar chart of `metric` ("Pearson r" or "RMSE") for a given missing level.
    Aggregates across all tissues (no tissue grouping).
    The best model is annotated with its marker symbol.

    Args:
        - df (DataFrame): full performance table (may span multiple miss levels)
        - metric (str): column to plot
        - miss_level (int): which miss level to filter / label in the filename
        - plot_dir (Path):
        - yticks (list|None): custom y-tick positions
    """
    # Filter to the requested missing level if the column exists
    if "Missing level" in df.columns:
        df = df[df["Missing level"] == miss_level].copy()
    else:
        df = df.copy()

    # Aggregate across all tissues - compute mean metric per model
    agg_df = df.groupby("Model")[metric].mean().reset_index()
    agg_df = agg_df.sort_values(metric, ascending=(metric == "RMSE"))

    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    x = np.arange(len(agg_df)) * 0.2
    y = agg_df[metric]

    bars = ax.bar(
        x,
        y,
        width=0.1,
        color=plt.cm.viridis(np.linspace(0, 1, len(x)))
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agg_df["Model"], ha="center", fontsize=9)

    # Annotate values above bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=8,
                    xytext=(0, 3),
                    textcoords='offset points')


    plt.xticks(ha="center", fontsize=9)
    if yticks:
        plt.yticks(yticks)
    plt.ylabel("Pearson\u2019s r" if metric == "Pearson r" else metric)
    plt.xlabel("")

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    metric_slug = metric.lower().replace(" ", "_").replace("\u2019", "")
    path = plot_dir / f"{metric_slug}_by_model_miss{miss_level}.png"
    plt.savefig(path)
    plt.show()
    print(f"Saved: {path}")

# =================================================
# =              2.1 Tissue                       =
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
        - tissue_performance (pd.DataFrame): DataFrame with columns ["Tissue", "Model", metric].
        - tissue_order: Ordered sequence of tissue labels matching the bar order in the plot.
        - metric (str): Metric column used to determine the best model per tissue.
        - best_fn: Function applied to the pivoted metric DataFrame that returns the best
            model for each tissue. Uses .idxmax() for metrics where higher
            is better and .idxmin() for metrics where lower is better.
    """
    pivot_df = tissue_performance.pivot(index="Tissue", columns="Model", values=metric)
    best_model_per_tissue = best_fn(pivot_df).to_dict()
    for container, model_name in zip(ax.containers, tissue_performance["Model"].unique()):
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
    # reference_model="Tissue Mean", 
    yticks: list=None
) -> None:
    """
    Bar chart of `metric` ("Pearson r" or "RMSE") per tissue for a given missing level.
    Tissues are sorted by number of samples (ascending).
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
    if "Missing level" in tissue_performance.columns:
        df = tissue_performance[tissue_performance["Missing level"] == miss_level].copy()
    else:
        df = tissue_performance.copy()

    order_df = (df[df["Tissue"].unique()].sort_values("Number of samples"))

    # order_df = (
    #     df[df["Model"] == reference_model]
    #     .sort_values("Number of samples")
    # )
    tissue_order = order_df["Tissue"].tolist()
    x_labels = [
        f"{t} (N = {n})"
        for t, n in zip(order_df["Tissue"], order_df["Number of samples"])
    ]

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    sns.barplot(data=df, x="Tissue", y=metric, hue="Model (Marker)",
                order=tissue_order, ax=ax)

    plt.xticks(range(len(x_labels)), x_labels, ha="center", fontsize=9)
    if yticks:
        plt.yticks(yticks)
    plt.ylabel("Pearson\u2019s r" if metric == "Pearson r" else metric)
    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.legend(title="Model", loc="lower left")

    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    best_fn = (lambda p: p.idxmax(axis=1)) if metric == "Pearson r" else (lambda p: p.idxmin(axis=1))
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

    n_tissues = len(df["Tissue"].unique())
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

    for i, tissue in enumerate(df["Tissue"].unique()):
        ax = axes[i]
        tissue_df = plot_df[plot_df["Tissue"] == tissue]
        num_samples = tissue_df["Number of samples"].iloc[0]

        palette = sns.color_palette()
        model_colors = {
            model: palette[i] 
            for i, model in enumerate(tissue_df["Model"].unique())
        }

        sns.scatterplot(
            data=tissue_df,
            x="True values",
            y="Predicted values",
            hue="Model",
            alpha=0.5,
            s=25,
            edgecolor="none",
            ax=ax
        )
        for model_name, model_df in tissue_df.groupby("Model"):
            sns.kdeplot(
                x=model_df["True values"],
                y=model_df["Predicted values"],
                ax=ax,
                levels=4,
                color=model_colors[model_name],
            )

        true_values = tissue_df["True values"]
        predicted_values = tissue_df["Predicted values"]
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