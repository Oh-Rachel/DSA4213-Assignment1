"""Neural language models: RNN, LSTM, and Transformer.

Both models are used the same way:
    logits, state = model(input_ids)
    - input_ids: token ids, shape [batch, length]
    - logits:    a score for every vocabulary word at every position
    - state:     the RNN/LSTM memory (None for the Transformer)

RecurrentLM (RNN or LSTM) is adapted from the DSA4213 Lecture 3 demo.
TransformerLM is new; its causal mask stops each position seeing later tokens.
"""
import torch
from torch import nn

EMBEDDING_DIM = 100  # matches GloVe 100d so Part I models can be reused in Part II


class RecurrentLM(nn.Module):
    """Vanilla RNN (cell="rnn") or LSTM (cell="lstm") language model."""

    is_recurrent = True

    def __init__(self, vocab_size, cell="lstm", embedding_dim=EMBEDDING_DIM,
                 hidden_size=256, num_layers=1, dropout=0.0, pad_id=0):
        super().__init__()
        if cell not in ("rnn", "lstm"):
            raise ValueError("cell must be 'rnn' or 'lstm'")
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        recurrent_class = nn.RNN if cell == "rnn" else nn.LSTM
        self.recurrent = recurrent_class(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def hidden_states(self, input_ids, state=None):
        """Top-layer hidden state at every position: [batch, length, hidden_size]."""
        x = self.dropout(self.embedding(input_ids))
        states, final_state = self.recurrent(x, state)
        return states, final_state

    def forward(self, input_ids, state=None):
        states, final_state = self.hidden_states(input_ids, state)
        return self.output(self.dropout(states)), final_state


class TransformerLM(nn.Module):
    """Decoder-only Transformer with learned positional embeddings and a causal mask."""

    is_recurrent = False

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, num_heads=4,
                 num_layers=2, feedforward_dim=256, max_length=64,
                 dropout=0.0, pad_id=0):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=feedforward_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def hidden_states(self, input_ids, padding_mask=None):
        """Final-layer representation at every position: [batch, length, embedding_dim].

        padding_mask: optional BoolTensor [batch, length], True where input is <pad>.
        """
        length = input_ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.max_length}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.embedding(input_ids) + self.position_embedding(positions))
        # Causal mask: position t may attend only to positions <= t.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=input_ids.device)
        x = self.blocks(x, mask=causal_mask, is_causal=True,
                        src_key_padding_mask=padding_mask)
        return self.norm(x)

    def forward(self, input_ids, state=None):
        return self.output(self.hidden_states(input_ids)), None


def set_embeddings(model, weights, freeze):
    """Copy a [vocab_size, embedding_dim] matrix into the input embeddings (Part II).

    freeze=True keeps the entire input-embedding table fixed during training.
    """
    weights = torch.as_tensor(weights, dtype=model.embedding.weight.dtype)
    if weights.shape != model.embedding.weight.shape:
        raise ValueError(f"expected {tuple(model.embedding.weight.shape)}, got {tuple(weights.shape)}")
    with torch.no_grad():
        model.embedding.weight.copy_(weights)
    model.embedding.weight.requires_grad = not freeze
    return model
