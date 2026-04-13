import logging

from analysis.plotter import Plotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_plot_pipeline(plot_type: str, run_dir: str) -> None:
    plotter = Plotter(run_dir)
    
    if plot_type == "training":
        plotter.plot_training()
    if plot_type == "evaluation":
        plotter.plot_evaluation()
    if plot_type == "latent":
        plotter.plot_latent()