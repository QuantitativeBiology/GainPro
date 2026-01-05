import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import logging
import os
from datetime import datetime

from gainpro.dann_utils import inverse_transform_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Correlation between the measured and predicted

# the correlation should only be computed on features that have a ground truth

def rmse_function(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))

def correlation_measured_predicted_sample(measured_sample, predicted_sample) -> tuple:
    """
        Computes the correlation, Pearson and RMSE, between the measured sample and the corresponding prediction.
    """
    
    # filter: in order to consider only the components with a ground truth associated
    x_filtered = [] # components of the measured sample different than NaN
    y_filtered = [] # components of the predicted sample different than NaN

    for i in range(len(measured_sample)):
        if not np.isnan(measured_sample[i]):
            x_filtered.append(measured_sample[i].item())
            y_filtered.append(predicted_sample[i].item())

    corr = np.corrcoef(x_filtered, y_filtered)[0,1].item()
    rmse = rmse_function(x_filtered, y_filtered).item()

    return (corr, rmse)

def correlation_measured_predicted(scaler, dataset, dataset_imputed, samples_names: list, sample_to_project: dict, save_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # todo do we want row-wise or column-wise?
    logger.info("\nPerforming the correlation analysis...")

    # scale (normalization) back
    dataset_tensor = inverse_transform_output(dataset, scaler)
    dataset_imputed_tensor = inverse_transform_output(dataset_imputed, scaler)

    dataset_np = dataset_tensor.numpy()
    dataset_imputed_np = dataset_imputed_tensor.numpy()

    correlations = []
    rmses = []

    dataset_np_filtered = [] #dataset by rows, each row only has the indices where the value is different than nan
    dataset_imputed_np_filtered = []

    for x, y in zip(dataset_np, dataset_imputed_np): # iterate over the samples
        
        # calculate the correlation of values that we have a ground truth
        x_filtered = []
        y_filtered = []

        for i in range(len(x)):
            if not np.isnan(x[i]):
                x_filtered.append(x[i].item())
                y_filtered.append(y[i].item())

        dataset_np_filtered.append(x_filtered)
        dataset_imputed_np_filtered.append(y_filtered)                
        
        correlations.append(np.corrcoef(x_filtered, y_filtered)[0,1].item())
        rmses.append(rmse_function(x_filtered, y_filtered))
        
    # print(f"Correlations: {correlations}")
    # print(f"Correlations: {len(correlations)}")

    df_corr = pd.DataFrame({
        'sample': samples_names,
        'correlation': correlations,
        'rmse': rmses
    })
    df_corr['project'] = df_corr['sample'].map(sample_to_project)
    pearson_stats = df_corr.groupby('project')['correlation'].agg(['mean', 'std', 'count']).reset_index()
    rmse_stats = df_corr.groupby('project')['rmse'].agg(['mean', 'std', 'count']).reset_index()
    # print("Pearson correlation")
    # print(pearson_stats)
    # print("RMSE")
    # print(rmse_stats)

    # Plot lowest and highest correlation samples per project
    selected_sample_ids = []

    for project in df_corr['project'].unique():
        project_samples = df_corr[df_corr['project'] == project]
        
        # Get sample with lowest and highest correlation
        min_idx = project_samples['correlation'].idxmin()
        max_idx = project_samples['correlation'].idxmax()
        
        selected_sample_ids.extend([min_idx, max_idx])

    selected_sample_ids = list(set(selected_sample_ids))

    for sample_id in selected_sample_ids: # iterate over some samples
        sample_name = samples_names[sample_id]
        sample_name = str(sample_name) if sample_name is not isinstance(sample_name, str) else sample_name
        project = sample_name.split('-Sample')[0]

        sample_original = dataset_np_filtered[sample_id]
        sample_predicted = dataset_imputed_np_filtered[sample_id]
        pearson = correlations[sample_id]
        rmse = rmses[sample_id]
        mean_pearson_project = pearson_stats.loc[pearson_stats['project'] == project, 'mean'].iloc[0]
        plot_correlation_sample(sample_name, sample_original, sample_predicted, pearson, rmse, mean_pearson_project, save_dir)

    return pearson_stats, rmse_stats

def plot_correlation_sample(sample_name: str, 
                            sample_original, sample_predicted, 
                            pearson: float, rmse: float, mean_pearson_project: float,
                            save_dir: str):
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
                label=f"Pearson's r={pearson:.3f} \nRMSE={rmse:.2f} \nMean Pearson Project={mean_pearson_project:.3f}") # disclaimer: não é bem comparável entre amostras porque
                                                                                        # o tamanho do vetor é diferente (diferente missing values)
    plt.plot(x, y, color="black", linestyle="-")
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

    path = os.path.join(save_dir, "imgs")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/correlation-{sample_name}.png")
    plt.close()