import os
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch

logger = logging.getLogger(__name__)


class EvaluationTracker:
    """
    Tracks evaluation metrics across folds (e.g., Pearson correlation, RMSE).
    Complements MetricsTracker, which is epoch-level.
    """

    def __init__(self, projects, scaler, save_dir=None):
        self.projects = np.unique(projects)
        self.scaler = scaler
        self.save_dir = save_dir

        # DataFrames for accumulating fold stats
        self.pearson_stats_fold = pd.DataFrame({
            "project": self.projects,
            "sum of folds mean": np.zeros(len(self.projects), dtype=np.float64),
            "count": np.zeros(len(self.projects), dtype=int),
            "total folds": np.zeros(len(self.projects), dtype=int),
        })
        self.rmse_stats_fold = self.pearson_stats_fold.copy()

    def evaluate_fold(self, dataset, dataset_imputed, sample_names, sample_to_project):
        """
        Perform correlation analysis for one fold and accumulate results.
        Returns per-project Pearson and RMSE for the fold.
        """
        logger.info("\nPerforming the correlation analysis...")

        # --- inverse transform (scaler normalization back) ---
        dataset_tensor = self._inverse_transform_output(dataset)
        dataset_imputed_tensor = self._inverse_transform_output(dataset_imputed)

        dataset_np = dataset_tensor.numpy()
        dataset_imputed_np = dataset_imputed_tensor.numpy()

        correlations, rmses = [], []
        dataset_np_filtered, dataset_imputed_np_filtered = [], []

        for x, y in zip(dataset_np, dataset_imputed_np):  # iterate over samples
            x_filtered, y_filtered = [], []
            for i in range(len(x)):
                if not np.isnan(x[i]):
                    x_filtered.append(x[i].item())
                    y_filtered.append(y[i].item())
            dataset_np_filtered.append(x_filtered)
            dataset_imputed_np_filtered.append(y_filtered)

            correlations.append(np.corrcoef(x_filtered, y_filtered)[0, 1].item())
            rmses.append(self._rmse_function(x_filtered, y_filtered))

        # --- build per-sample dataframe ---
        df_corr = pd.DataFrame({
            "sample": sample_names,
            "correlation": correlations,
            "rmse": rmses,
        })
        df_corr["project"] = df_corr["sample"].map(sample_to_project)

        # --- aggregate per project ---
        pearson_stats = df_corr.groupby("project")["correlation"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        rmse_stats = df_corr.groupby("project")["rmse"].agg(
            ["mean", "std", "count"]
        ).reset_index()

        # --- accumulate into global stats ---
        self._accumulate(self.pearson_stats_fold, pearson_stats)
        self._accumulate(self.rmse_stats_fold, rmse_stats)

        # --- plots for extreme samples ---
        self._plot_extreme_samples(
            df_corr, sample_names, dataset_np_filtered, dataset_imputed_np_filtered,
            correlations, rmses, pearson_stats
        )

        return pearson_stats, rmse_stats

    def finalize(self):
        """
        Compute mean Pearson and RMSE across folds.
        """
        self.pearson_stats_fold["Pearson mean across folds"] = (
            self.pearson_stats_fold["sum of folds mean"] / self.pearson_stats_fold["total folds"]
        )
        self.rmse_stats_fold["RMSE mean across folds"] = (
            self.rmse_stats_fold["sum of folds mean"] / self.rmse_stats_fold["total folds"]
        )
        return self.pearson_stats_fold, self.rmse_stats_fold

    def plot(self):
        """
        Plot Pearson correlation and RMSE statistics for each project.
        """
        
        # Create a color map (distinct color per project)
        palette = sns.color_palette("husl", n_colors=len(self.projects), desat=0.55)
        color_dict = {proj: color for proj, color in zip(sorted(self.projects), palette)}

        # Pearson plot
        df = self.pearson_stats_fold.sort_values("Pearson mean across folds", ascending=False)

        _, ax = plt.subplots(figsize=(10,8))
        bar = plt.barh(
            df["project"], 
            df["Pearson mean across folds"], 
            align="center", 
            color=[color_dict[p] for p in df["project"]])
        
        for bar, value in zip(bar, df["Pearson mean across folds"].values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{value:.3f}", va='center', fontsize=8)
                
            new_labels = [
                f"{proj}\n n={cnt}"
                for proj, cnt in zip(df["project"], df["count"])
            ]

            ax.set_yticks(range(len(new_labels)))
            ax.set_yticklabels(new_labels)
        
        # ax.set_title(f"Average Pearson across folds by Project", loc="left")

        plt.grid(True, which='major', color='lightgray', linewidth=0.5, axis="x")
        ax.set_axisbelow(True)
        ax.invert_yaxis()  # highest Pearson at top
        plt.tick_params(axis='both', which='both', direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(self.save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/average-pearson-across-folds-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        plt.close()

        # RMSE plot
        df = self.rmse_stats_fold.sort_values("RMSE mean across folds", ascending=False)

        _, ax = plt.subplots(figsize=(10,8))
        bar = plt.barh(df["project"], df["RMSE mean across folds"],
            align="center", 
            color=[color_dict[p] for p in df["project"]])
        
        for bar, value in zip(bar, df["RMSE mean across folds"].values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{value:.3f}", va='center', fontsize=8)
                
            new_labels = [
                f"{proj}\n n={cnt}"
                for proj, cnt in zip(df["project"], df["count"])
            ]

            ax.set_yticks(range(len(new_labels)))
            ax.set_yticklabels(new_labels)
        
        # ax.set_title(f"Average RMSE across folds by Project", loc="left")
    
        plt.grid(True, which='major', color='lightgray', linewidth=0.5, axis="x")
        ax.set_axisbelow(True)
        plt.tick_params(axis='both', which='both', direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(self.save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/average-rmse-across-folds-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        plt.close()

    # -------------------- Helpers --------------------

    def _accumulate(self, stats_fold: pd.DataFrame, stats_current: pd.DataFrame):
        """
        Accumulate statistics across folds.
        
        Args:
            - stats_fold (pd.DataFrame): Accumulated statistics from previous folds.
            - stats_current (pd.DataFrame): Statistics from the current fold.
            
        Returns:
            - pd.DataFrame: Updated accumulated statistics.
        """
        for _, row in stats_current.iterrows():
            project = row["project"]
            if project in stats_fold["project"].values:
                # Update existing project
                stats_fold.loc[stats_fold["project"] == project, "sum of folds mean"] += row["mean"]
                stats_fold.loc[stats_fold["project"] == project, "count"] += row["count"]
                stats_fold.loc[stats_fold["project"] == project, "total folds"] += 1
            else:
                stats_fold.loc[project] = [row["mean"], row["count"]]
                stats_fold.loc[project]["total folds"] += 1

    def _plot_extreme_samples(self, df_corr, sample_names,
                            dataset_np_filtered, dataset_imputed_np_filtered,
                            correlations, rmses, pearson_stats):
        """
        Plot lowest and highest correlation samples per project.
        """

        selected_ids = []
        for project in df_corr["project"].unique():
            project_samples = df_corr[df_corr["project"] == project]

            # Get sample with lowest and highest correlation
            selected_ids.extend([
                project_samples["correlation"].idxmin(),
                project_samples["correlation"].idxmax(),
            ])

        selected_ids = list(set(selected_ids))

        for sample_id in selected_ids:
            sample_name = sample_names[sample_id]
            sample_name = str(sample_name) if sample_name is not isinstance(sample_name, str) else sample_name
            project = sample_name.split("-Sample")[0]

            sample_original = dataset_np_filtered[sample_id]
            sample_predicted = dataset_imputed_np_filtered[sample_id]
            pearson = correlations[sample_id]
            rmse = rmses[sample_id]
            mean_pearson_project = pearson_stats.loc[pearson_stats["project"] == project, "mean"].iloc[0]

            self._plot_correlation_sample(
                sample_name, sample_original, sample_predicted,
                pearson, rmse, mean_pearson_project
            )

    def _plot_correlation_sample(self, sample_name: str, sample_original, sample_predicted,
                                pearson: float, rmse: float, mean_pearson: float):
        
        # identity line
        xlim = 1000000
        x = np.linspace(-xlim, xlim, 100)
        y = x

        x_min = np.min(sample_original)
        x_max = np.max(sample_original)
        y_min = np.min(sample_predicted)
        y_max = np.max(sample_predicted)

        plt.figure(figsize=(10, 8))
        plt.scatter(sample_original, sample_predicted, color="#fc8b64", 
                    label=f"Pearson's r={pearson:.2f} \nRMSE={rmse:.2f} \nMean Pearson Project={mean_pearson:.2f}") # disclaimer: não é bem comparável entre amostras porque
                                                                                            # o tamanho do vetor é diferente (diferente missing values)
        plt.plot(x, y, color="gray", linestyle="-")
        # sns.lmplot(x='original', y='imputed', data=df, scatter_kws={'color':'orange'}, line_kws={'color':'black'})
        plt.legend(markerscale=0)
        plt.xlabel("Original")
        plt.ylabel("Imputed (GAIN-DANN)")
        plt.title(f"Original vs. Imputed values for sample {sample_name}", loc="left")
        plt.xlim(x_min*0.9, x_max*1.1)
        plt.ylim(y_min*0.9, y_max*1.1)
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

        path = os.path.join(self.save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/correlation-{sample_name}.png")
        plt.close()

    def _inverse_transform_output(self, imputed_data: torch.tensor) -> torch.Tensor:
        """
        Transforms the imputed data from standardized space back to the original proteomics scale.

        Args:
            - imputed_data (torch.Tensor): The imputed dataset in standardized space.
            - scaler (StandardScaler): The fitted scaler used for normalization.

        Returns:
            - torch.Tensor: The imputed data in the original proteomics scale.
        """
        imputed_data = self.scaler.inverse_transform(imputed_data)
        return torch.tensor(imputed_data, dtype=torch.float32)

    def _rmse_function(self, x, y):
        return np.sqrt(np.mean((np.array(x) - np.array(y)) ** 2))
