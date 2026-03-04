import time
import torch
import torch.nn as nn
import logging
from pathlib import Path
from torchviz import make_dot
import yaml

from models.GainDann.gain_dann import GAIN_DANN
from models.ProtoGain.dataset import generate_hint
from models.hypers import Hypers
from train.losses import compute_domain_loss, compute_gain_loss, compute_reconstruction_loss, compute_model_loss, compute_task_specific_loss, compute_imputation_validation
from train.early_stopping import EarlyStopping
from eval.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer():
    def __init__(
        cls, 
        model: GAIN_DANN, 
        hypers: Hypers,
        save_model: bool,
        run_dir: Path = None,
    ):
        cls.model = model
        cls.hypers = hypers
        cls.optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters()},
                {'params': model.domain_classifier.parameters()},
                {'params': model.decoder.parameters()}
                ],
                lr=hypers.learning_rate,
                weight_decay=hypers.weight_decay)
        cls.metrics = MetricsTracker()
        cls.save_model = save_model
        cls.run_dir = run_dir
    
    def fit(cls, train_loader, val_loader=None) -> None:
        torch.autograd.set_detect_anomaly(True)

        early_stopper = EarlyStopping(cls.hypers.early_stop_patience)
        epochs = cls.hypers.epochs
        start_time = time.time()
        for ep in range(1, epochs+1):
            logger.info(f"\nEpoch {ep}/{epochs}")

            cls._train_epoch(train_loader)
            if val_loader is not None:
                imputation_val = cls._validation_epoch(val_loader)
                logger.info(f"[Validation] Imputation epoch: {imputation_val}")
                if early_stopper.step(imputation_val, ep):
                    logger.info(f"Early stopping at epoch {ep}")
            
            cls.metrics.finish_epoch()
            print("Metrics epoch", cls.metrics._train_metrics_ep)

        logger.info(f"Training time: {time.time() - start_time} seconds")
        logger.info("Training done!")

        if cls.save_model:
            cls.save_checkpoint(cls.run_dir)
    
    def _train_epoch(cls, train_loader, lambd: float=1.0) -> None:
        cls.model.train()

        # decoder_a = cls.model.decoder.state_dict()["layers.0.weight"].clone().detach()
        # a = a[:3,:4]
        for batch in train_loader:
            x, x_missing, y, mask = batch

            z, z_imputed_aux, domain_logits, x_imputed = cls.model(x_missing, mask, lambd=lambd)
            domain_classifier_loss = compute_domain_loss(domain_logits, y)
            domain_pred = torch.argmax(domain_logits, dim=1)
            domain_accuracy = (domain_pred == y).float().mean()
            
            n_samples, dim = z.shape[0], z.shape[1]
            Z = torch.rand((n_samples, dim)) * 0.01
            hint = generate_hint(mask, cls.hypers.hint_rate)
            cls.model.gain._update_D(z.clone().detach(), mask, hint, Z, nn.BCELoss(reduction="none"))
            cls.model.gain._update_G(z.clone().detach(), mask, hint, Z, nn.BCELoss(reduction="none"))
            gain_loss = compute_gain_loss(z_imputed_aux, z.clone().detach(), mask)

            decoder_loss = compute_reconstruction_loss(x_imputed, x, mask)
            # print("Decoder loss", decoder_loss)
            # print("\n\n")
            # print("X", x)
            # print("X recon", x_imputed)
            # print("\n\n")

            task_specific_loss = compute_task_specific_loss(
                gain_loss=gain_loss.item(),
                reconstruction_loss=decoder_loss.item(),
                alpha=cls.hypers.alpha_weight,
                beta=cls.hypers.beta_weight,
            )

            model_loss = compute_model_loss(
                gain_loss=gain_loss.item(),
                reconstruction_loss=decoder_loss,
                domain_classifier_loss=domain_classifier_loss.mean(),
                alpha=cls.hypers.alpha_weight,
                beta=cls.hypers.beta_weight,
                gamma=cls.hypers.gamma_weight
            )
            logger.info(f"Gain loss {gain_loss.item()} | Domain loss {domain_classifier_loss.mean()} | Decoder loss {decoder_loss.item()} | Model loss {model_loss}")

            cls.metrics.update_train(gain_loss=gain_loss.item(),
                domain_classifier_loss=domain_classifier_loss.mean().item(),
                domain_accuracy=domain_accuracy.item(),
                decoder_loss=decoder_loss.item(),
                task_specific_loss=task_specific_loss,
                model_loss=model_loss.item()
            )

            cls.optimizer.zero_grad()
            model_loss.backward()
            cls.optimizer.step()
        
        # decoder_b = cls.model.decoder.state_dict()["layers.0.weight"].clone().detach()
        # b = b[:3,:4]
        # print(a)
        # print("\n\n")
        # print(b)
        # print("\n\n")
        # print(decoder_a.equal(decoder_b))
    
    def _validation_epoch(cls, val_loader, lambd: float=1.0):
        cls.model.eval()

        imputation_val_epoch = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x, x_missing, y, mask = batch
                missing_mask = torch.logical_and(torch.isnan(x_missing), ~torch.isnan(x))

                z, z_imputed_aux, domain_logits, x_imputed = cls.model(x_missing, mask, lambd=lambd)
                domain_classifier_loss = compute_domain_loss(domain_logits, y)
                domain_pred = torch.argmax(domain_logits, dim=1)
                domain_accuracy = (domain_pred == y).float().mean()
                
                gain_loss = compute_gain_loss(z_imputed_aux, z.clone().detach(), mask)

                decoder_loss = compute_reconstruction_loss(x_imputed, x, mask)

                task_specific_loss = compute_task_specific_loss(
                    gain_loss=gain_loss.item(),
                    reconstruction_loss=decoder_loss.item(),
                    alpha=cls.hypers.alpha_weight,
                    beta=cls.hypers.beta_weight,
                )

                model_loss = compute_model_loss(
                    gain_loss=gain_loss.item(),
                    reconstruction_loss=decoder_loss,
                    domain_classifier_loss=domain_classifier_loss.mean(),
                    alpha=cls.hypers.alpha_weight,
                    beta=cls.hypers.beta_weight,
                    gamma=cls.hypers.gamma_weight
                )

                imputation_val = compute_imputation_validation(x_imputed, x, missing_mask)
                imputation_val = imputation_val
                logger.info(f"Validation imputation {imputation_val}")
                imputation_val_epoch += imputation_val

                logger.info(f"[Val] Gain loss {gain_loss.item()} | Domain loss {domain_classifier_loss.mean()} | Decoder loss {decoder_loss.item()} | Model loss {model_loss}")

                cls.metrics.update_val(gain_loss=gain_loss.item(),
                    domain_classifier_loss=domain_classifier_loss.mean().item(),
                    domain_accuracy=domain_accuracy.item(),
                    decoder_loss=decoder_loss.item(),
                    task_specific_loss=task_specific_loss,
                    model_loss=model_loss.item()
                )

        n_batches = len(val_loader)
        return imputation_val_epoch / n_batches
    
    def save_checkpoint(
        cls, 
        run_dir: Path = None
    ) -> None:
        if run_dir is None:
            run_dir = cls.run_dir

        model_path = Path(f"{run_dir}/model.pt")
        torch.save(cls.model.state_dict(), model_path)

        metadata_path = Path(f"{run_dir}/metadata.yaml")

        # Convert non-serializable items
        serializable_metadata = {}
        hypers = cls.hypers.to_dict()
        for key, value in hypers.items():
            if hasattr(value, "dict"):
                serializable_metadata[key] = value.dict()
            elif isinstance(value, (int, float, str, list, dict, bool, type(None))):
                serializable_metadata[key] = value
            else:
                serializable_metadata[key] = str(value)

        metadata = {
            "input_dim": cls.model.get_input_dim(), 
            "latent_dim": cls.model.get_latent_dim(),
            "n_class": cls.model.get_n_class(),
            "hypers": serializable_metadata,
            "protein_names": cls.model.get_protein_names()
        }
        
        with open(metadata_path, "w") as f:
            yaml.safe_dump(
                metadata,
                f,
                default_flow_style=False,
                sort_keys=False
            )
