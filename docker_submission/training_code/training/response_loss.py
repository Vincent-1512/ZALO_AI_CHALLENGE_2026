"""Causal SFT loss with vocabulary logits allocated only for supervised positions."""
from torch.utils.data import SequentialSampler
import torch.nn.functional as F
from transformers import Trainer


def response_only_loss(model, inputs):
    targets = inputs['labels'][:, 1:]
    positions = (targets != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if not positions.numel():
        raise ValueError('Batch has no supervised next-token targets')
    model_inputs = {key: value for key, value in inputs.items() if key != 'labels'}
    outputs = model(**model_inputs, logits_to_keep=positions, use_cache=False)
    selected_targets = targets.index_select(1, positions)
    loss = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                           selected_targets.reshape(-1), ignore_index=-100)
    return loss, outputs


class ResponseOnlyTrainer(Trainer):
    def __init__(self, *args, longest_first=False, **kwargs):
        self.longest_first = longest_first
        super().__init__(*args, **kwargs)
        # compute_loss returns a per-microbatch mean; Trainer handles accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset=None):
        if self.longest_first:
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = response_only_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss
