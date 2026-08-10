"""Datasets and model architectures, defined once.

Every notebook — hyperparameter search, cross-validation and the final test
evaluation — imports its models from here:

    sys.path.insert(0, "<path to v2_heldout>")
    from config import *
    from models import *

That matters more than it looks. The central claim of this work rests on the
architectures being *identical* across the search, the cross-validation and the
test evaluation: the fold weights saved by a CV notebook are loaded again by
`test_evaluation_v2`, so any drift between two copies of a class either breaks
`load_state_dict` or, worse, silently scores a different model. Keeping one
definition makes that drift impossible rather than merely unlikely.

Note for Colab: this module is cached on first import. After editing it, restart
the runtime (or `importlib.reload(models)`) before re-running a notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification

try:                       # keep the sequence length in one place as well
    from config import MAX_LEN
except ImportError:        # stand-alone use
    MAX_LEN = 128

__all__ = [
    "MAX_LEN",
    "SurfaceOnlyDataset",
    "DualViewSurfaceLemmaDataset",
    "dual_collator",
    "dual_surface_lemma_collator",
    "CollapseError",
    "split_param_groups",
    "DualViewCNNBiLSTMAttention",
]


# --------------------------------------------------------------- datasets
class SurfaceOnlyDataset(Dataset):
    """Surface view only; the lemma column is never read."""

    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts     = df["surface"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DualViewSurfaceLemmaDataset(Dataset):
    """Surface and second view, tokenised separately and aligned by index.

    `lemma_col` selects the column feeding the second branch. It exists for the
    capacity-matched control `dual_view_dupinput_v2`, which passes
    ``lemma_col="surface"``: the model, its parameters and its configuration are
    then identical to the dual-view model, and only the information carried by
    the second view differs.
    """

    def __init__(self, df, tokenizer, max_len=MAX_LEN, lemma_col="lemma"):
        self.surface   = df["surface"].tolist()
        self.lemma     = df[lemma_col].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def encode(self, text):
        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx):
        return {
            "surface": self.encode(self.surface[idx]),
            "lemma":   self.encode(self.lemma[idx]),
            "labels":  torch.tensor(self.labels[idx], dtype=torch.long),
        }


def dual_collator(features):
    """Batches the nested surface/lemma structure of the dual-view dataset."""
    batch = {}
    for key in ["surface", "lemma"]:
        batch[key] = {
            "input_ids":      torch.stack([f[key]["input_ids"]      for f in features]),
            "attention_mask": torch.stack([f[key]["attention_mask"] for f in features]),
        }
    batch["labels"] = torch.stack([f["labels"] for f in features])
    return batch


# the notebooks historically used the longer name; keep both working
dual_surface_lemma_collator = dual_collator


# ---------------------------------------------------------------- training
class CollapseError(RuntimeError):
    """Raised when a fine-tuning run diverges (epoch-1 loss above ln(3))."""


def split_param_groups(model, encoder_attr="encoder"):
    """Split a model into (pretrained encoder, randomly initialised head).

    A newly initialised head needs a considerably higher learning rate than the
    pretrained body (discriminative fine-tuning, Howard and Ruder, 2018). Every
    architecture uses this same split, so no variant trains its head at the
    encoder rate by accident.

    Works for both the custom architectures below, whose encoder lives in
    `model.encoder`, and for `AutoModelForSequenceClassification`, where it is
    named by `model.base_model_prefix` (``roberta`` for XLM-R).
    """
    prefix = getattr(model, "base_model_prefix", None) or encoder_attr
    if not hasattr(model, prefix):
        prefix = encoder_attr
    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith(prefix + ".")]
    head_params    = [p for n, p in model.named_parameters()
                      if not n.startswith(prefix + ".")]
    assert encoder_params, f"no encoder parameters found under '{prefix}.'"
    assert head_params, f"no head parameters found outside '{prefix}.'"
    return encoder_params, head_params


# ------------------------------------------------------------ architectures
class DualViewCNNBiLSTMAttention(nn.Module):
    """Dual-view SARF adaptation: surface + lemma, fused, CNN-BiLSTM head.

    `gated=False` fuses by averaging (Eq. 1 in the paper); `gated=True` learns a
    per-token gate over the two views (Eq. 2) and additionally returns those gate
    weights, which is what the gate-saturation analysis reads.

    Note that the CNN branch always reads the *surface* representation; only the
    BiLSTM branch sees the fusion. The second view therefore reaches 256 of the
    856 head features, which bounds how much it can contribute by construction.
    """

    def __init__(self, bert_model_name="xlm-roberta-large", lstm_hidden_dim=128,
                 cnn_filters=200, kernel_sizes=(3, 4, 5), num_classes=3,
                 dropout=0.3, gated=False):
        super(DualViewCNNBiLSTMAttention, self).__init__()
        self.gated = gated

        base_model = AutoModelForSequenceClassification.from_pretrained(
            bert_model_name, return_dict=True, num_labels=num_classes
        )
        self.encoder = (base_model.roberta if hasattr(base_model, "roberta")
                        else base_model.bert)
        bert_hidden_dim = self.encoder.config.hidden_size

        self.cross_attn_lemma = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim, num_heads=8, batch_first=True
        )
        if gated:
            self.fusion_gate = nn.Linear(bert_hidden_dim * 2, 1)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=cnn_filters,
                      kernel_size=(k, bert_hidden_dim))
            for k in kernel_sizes
        ])
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_dim, hidden_size=lstm_hidden_dim,
            num_layers=1, bidirectional=True, batch_first=True
        )

        fused_dim = (cnn_filters * len(kernel_sizes)) + (lstm_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(fused_dim, num_classes)

    def forward(self, surface, lemma):
        surface_mask = surface["attention_mask"]
        lemma_mask   = lemma["attention_mask"]

        surface_out = self.dropout(self.encoder(**surface).last_hidden_state)
        lemma_out   = self.dropout(self.encoder(**lemma).last_hidden_state)

        lemma_attn, _ = self.cross_attn_lemma(
            query=surface_out, key=lemma_out, value=lemma_out,
            key_padding_mask=(lemma_mask == 0)
        )

        if self.gated:
            concat_features = torch.cat((surface_out, lemma_attn), dim=-1)
            surface_weight  = torch.sigmoid(self.fusion_gate(concat_features))
            cross_attended  = ((surface_weight * surface_out)
                               + ((1.0 - surface_weight) * lemma_attn))
        else:
            surface_weight = None
            cross_attended = (surface_out + lemma_attn) / 2.0

        pad_mask_2d    = (surface_mask == 0).unsqueeze(-1)
        cross_attended = self.dropout(cross_attended.masked_fill(pad_mask_2d, 0.0))

        pad_mask_4d = (surface_mask == 0).unsqueeze(1).unsqueeze(-1)
        cnn_input   = surface_out.unsqueeze(1).masked_fill(pad_mask_4d, 0.0)

        cnn_features = []
        for conv in self.convs:
            x = F.relu(conv(cnn_input)).squeeze(3)
            x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            cnn_features.append(x)
        cnn_out = torch.cat(cnn_features, dim=1)

        lengths      = surface_mask.sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            cross_attended, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed_input)
        lstm_out, _      = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=surface_mask.size(1)
        )

        input_mask_expanded = surface_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        sum_embeddings = torch.sum(lstm_out * input_mask_expanded, dim=1)
        sum_mask       = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        lstm_pooled    = sum_embeddings / sum_mask

        fused  = torch.cat((cnn_out, lstm_pooled), dim=1)
        logits = self.fc(self.dropout(fused))
        return (logits, surface_weight) if self.gated else logits
