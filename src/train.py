"""Training loop shared by every neural language model (Parts I and II).

Training data is cut into non-overlapping chunks of seq_len tokens with
one-step-shifted targets, as in sequence_chunks() from the DSA4213 Lecture 3
demo. Chunks are shuffled each epoch, and every chunk starts from a fresh
state, so recurrent and Transformer models see exactly the same examples.

Changes from the Lecture 3 training loop:
    - a validation loss is recorded after every epoch, for loss curves
    - Adam is the default optimiser
    - the weights with the lowest validation loss are restored at the end.
"""
import copy
import time

import numpy as np
import torch
from torch import nn


def make_chunks(ids, seq_len):
    """Non-overlapping (input, target) chunks: two LongTensors [num_chunks, seq_len]."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    num_chunks = (len(ids) - 1) // seq_len
    usable = num_chunks * seq_len
    inputs = ids[:usable].view(num_chunks, seq_len)
    targets = ids[1 : usable + 1].view(num_chunks, seq_len)
    return inputs, targets


def chunked_loss(model, inputs, targets, loss_function, batch_size):
    """Mean token loss over pre-made chunks, without updating the model."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            logits, _ = model(batch_inputs)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1))
            tokens = (batch_targets != loss_function.ignore_index).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens


def train_language_model(model, train_ids, valid_ids, *, epochs=10, seq_len=64,
                         batch_size=32, learning_rate=1e-3, clip=1.0,
                         pad_id=0, seed=4213, verbose=True):
    """Train with teacher forcing and return a history dict.

    history keys: train_loss, valid_loss (per epoch), epoch_seconds,
    total_seconds, best_epoch (1-based), seq_len, batch_size, learning_rate.
    """
    torch.manual_seed(seed)
    shuffle_rng = np.random.default_rng(seed)

    train_inputs, train_targets = make_chunks(train_ids, seq_len)
    valid_inputs, valid_targets = make_chunks(valid_ids, seq_len)
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    history = {"train_loss": [], "valid_loss": [], "epoch_seconds": [],
               "seq_len": seq_len, "batch_size": batch_size,
               "learning_rate": learning_rate}
    best_valid, best_state = float("inf"), None
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = shuffle_rng.permutation(len(train_inputs))
        total_loss, total_tokens = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_inputs, batch_targets = train_inputs[batch], train_targets[batch]
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            optimizer.step()
            tokens = (batch_targets != pad_id).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

        train_loss = total_loss / total_tokens
        valid_loss = chunked_loss(model, valid_inputs, valid_targets, loss_function, batch_size)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["epoch_seconds"].append(time.perf_counter() - epoch_start)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        if verbose:
            print(f"epoch {epoch:>2}: train {train_loss:.4f}  valid {valid_loss:.4f}  "
                  f"({history['epoch_seconds'][-1]:.1f}s)")

    history["total_seconds"] = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    return history
