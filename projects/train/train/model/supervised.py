import torch
from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase

Tensor = torch.Tensor


class SupervisedAframe(AframeBase):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        return self.model(X)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y = batch
        y_hat = self(X)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        return self(X)


class StateSpaceAframe(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # scale lr by number of GPUs
        # https://arxiv.org/pdf/1706.02677.pdf
        lr = self.hparams.learning_rate * world_size
        self._logger.info(f"Scaled lr by {world_size} to {lr}")
        optimizer = torch.optim.AdamW(
            params, lr, weight_decay=self.hparams.weight_decay
        )

        # Add parameters with special hyperparameters
        hps = [
            getattr(p, "_optim")
            for p in all_parameters
            if hasattr(p, "_optim")
        ]
        hps = [
            dict(s)
            for s in sorted(
                list(dict.fromkeys(frozenset(hp.items()) for hp in hps))
            )
        ]  # Unique dicts
        for hp in hps:
            params = [
                p for p in all_parameters if getattr(p, "_optim", None) == hp
            ]
            optimizer.add_param_group({"params": params, **hp})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )

        scheduler_config = dict(scheduler=scheduler, interval="step")
        return dict(optimizer=optimizer, lr_scheduler=scheduler_config)
