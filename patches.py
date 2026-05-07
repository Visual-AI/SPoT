"""
Monkey-patch TRL's DPOTrainer to free the reference model from GPU
after precomputing reference log probabilities.

Import this module before creating any DPOTrainer instance:
    import patches  # noqa: F401
"""

import gc

from trl.trainer.dpo_trainer import DPOTrainer, empty_cache

_original_get_train_dataloader = DPOTrainer.get_train_dataloader


def _get_train_dataloader_with_free_ref(self):
    dataloader = _original_get_train_dataloader(self)

    if (
        self.precompute_ref_log_probs
        and self._precomputed_train_ref_log_probs
        and self.ref_model is not None
    ):
        print("[patches] Precompute done. Freeing reference model from memory...")
        del self.ref_model
        self.ref_model = None
        gc.collect()
        empty_cache()
        self.accelerator.free_memory()
        print("[patches] Reference model freed.")

    return dataloader


DPOTrainer.get_train_dataloader = _get_train_dataloader_with_free_ref
