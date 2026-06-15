import torch

from src.metrics.analysis_metrics import collect_epoch_analysis_metrics
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer for language-modeling experiments.
    """

    def process_batch(self, batch, metrics: MetricTracker, zero_grad=True, update=True):
        batch = self.move_batch_to_device(batch)
        if self.is_train and zero_grad:
            self.optimizer.zero_grad(set_to_none=True)

        with self.autocast_context:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            scaled_loss = batch["loss"] / self.accumulation_steps
            self.autocast_grad_scaler.scale(scaled_loss).backward()
            if update:
                self._clip_grad_norm()
                self.autocast_grad_scaler.step(self.optimizer)
                self.autocast_grad_scaler.update()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        metrics_split = "train" if self.is_train else "inference"
        for met in self.metrics[metrics_split]:
            if hasattr(met, "reset") and metrics_split == "inference":
                pass
            metrics.update(met.name, met(**batch))

        return batch

    def _append_analysis_metrics(self, logs, epoch, part=None):
        analysis_logs = collect_epoch_analysis_metrics(self._unwrap_model(), self.config)
        for metric_name, value in analysis_logs.items():
            logs[metric_name] = value
            scalar_name = f"{part}_{metric_name}" if part else metric_name
            self.writer.add_scalar(scalar_name, value, epoch)

    def _train_epoch(self, epoch):
        logs = super()._train_epoch(epoch)
        self._append_analysis_metrics(logs, epoch)
        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        for met in self.metrics["inference"]:
            if hasattr(met, "reset"):
                met.reset()

        val_logs = super()._evaluation_epoch(epoch, part, dataloader)

        for met in self.metrics["inference"]:
            if hasattr(met, "value"):
                val_logs[met.name] = met.value()

        self._append_analysis_metrics(val_logs, epoch, part=part)
        return val_logs

    def _unwrap_model(self):
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        return model

    def _log_batch(self, batch_idx, batch, mode="train"):
        return
