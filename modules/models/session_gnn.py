"""Action-aware SessionGNN ranker for the HRFlow hybrid pipeline."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


class _SessionGNNTrainingDataset(Dataset):
    """Lightweight dataset wrapper over precomputed training examples."""

    def __init__(self, examples: List[Dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def _build_session_graph(
    prefix_item_idxs: List[int],
    prefix_action_idxs: List[int],
    use_action_weighting: bool,
    action_edge_weights: Dict[int, float],
) -> Dict[str, Any]:
    """Build the directed session graph used by SR-GNN."""
    if len(prefix_item_idxs) != len(prefix_action_idxs):
        raise ValueError(
            "prefix_item_idxs and prefix_action_idxs must have equal length"
        )

    node_item_ids: List[int] = []
    item_to_node: Dict[int, int] = {}
    alias: List[int] = []

    for it in prefix_item_idxs:
        if it not in item_to_node:
            item_to_node[it] = len(node_item_ids)
            node_item_ids.append(int(it))
        alias.append(item_to_node[it])

    edge_count: Dict[tuple[int, int], float] = {}
    for t, (src, dst) in enumerate(zip(alias[:-1], alias[1:]), start=1):
        dst_action = int(prefix_action_idxs[t])
        w = (
            float(action_edge_weights.get(dst_action, 1.0))
            if use_action_weighting
            else 1.0
        )
        edge_count[(src, dst)] = edge_count.get((src, dst), 0.0) + w

    if edge_count:
        edge_src = np.array([k[0] for k in edge_count.keys()], dtype=np.int64)
        edge_dst = np.array([k[1] for k in edge_count.keys()], dtype=np.int64)
        edge_w = np.array(list(edge_count.values()), dtype=np.float32)
    else:
        edge_src = np.zeros((0,), dtype=np.int64)
        edge_dst = np.zeros((0,), dtype=np.int64)
        edge_w = np.zeros((0,), dtype=np.float32)

    return {
        "node_item_ids": node_item_ids,
        "alias": alias,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_w": edge_w,
    }


def _collate_sessiongnn_examples(
    batch: List[Dict[str, Any]],
    max_seq_len: int,
    use_action_weighting: bool,
    action_edge_weights: Dict[int, float],
    default_action_idx: int,
) -> Dict[str, Any]:
    """Collate examples into a batched graph structure."""
    all_node_item_ids: List[int] = []
    all_edge_src: List[int] = []
    all_edge_dst: List[int] = []
    all_edge_w: List[float] = []

    alias_inputs: List[List[int]] = []
    seq_mask: List[List[float]] = []
    seq_action_ids: List[List[int]] = []
    lengths: List[int] = []
    pos_items: List[int] = []
    history_sets: List[set[int]] = []
    graph_node_ptr = [0]

    node_offset = 0

    for row in batch:
        prefix = list(row["prefix_item_idxs"][-max_seq_len:])
        prefix_actions = list(
            row.get("prefix_action_idxs", [default_action_idx] * len(prefix))
        )[-max_seq_len:]

        if len(prefix_actions) < len(prefix):
            prefix_actions = [default_action_idx] * (
                len(prefix) - len(prefix_actions)
            ) + prefix_actions
        elif len(prefix_actions) > len(prefix):
            prefix_actions = prefix_actions[-len(prefix) :]

        g = _build_session_graph(
            prefix_item_idxs=[int(x) for x in prefix],
            prefix_action_idxs=[int(x) for x in prefix_actions],
            use_action_weighting=use_action_weighting,
            action_edge_weights=action_edge_weights,
        )

        n_nodes = len(g["node_item_ids"])
        all_node_item_ids.extend(
            [x + 1 for x in g["node_item_ids"]]
        )  # +1 padding shift

        if len(g["edge_src"]) > 0:
            all_edge_src.extend((g["edge_src"] + node_offset).tolist())
            all_edge_dst.extend((g["edge_dst"] + node_offset).tolist())
            all_edge_w.extend(g["edge_w"].tolist())

        alias = g["alias"][-max_seq_len:]
        length = len(alias)
        pad = max_seq_len - length

        alias_inputs.append([0] * pad + [a + node_offset for a in alias])
        seq_mask.append([0.0] * pad + [1.0] * length)
        seq_action_ids.append([0] * pad + [int(a) for a in prefix_actions])
        lengths.append(length)
        pos_items.append(int(row.get("pos_item_idx", 0)))
        history_sets.append(set(prefix))

        node_offset += n_nodes
        graph_node_ptr.append(node_offset)

    if len(all_node_item_ids) == 0:
        # Keep one padding node so tensor indexing remains valid.
        all_node_item_ids = [0]
        graph_node_ptr = [0, 1]

    return {
        "node_item_ids": torch.tensor(all_node_item_ids, dtype=torch.long),
        "edge_src": torch.tensor(all_edge_src, dtype=torch.long),
        "edge_dst": torch.tensor(all_edge_dst, dtype=torch.long),
        "edge_w": torch.tensor(all_edge_w, dtype=torch.float32),
        "graph_node_ptr": torch.tensor(graph_node_ptr, dtype=torch.long),
        "alias_inputs": torch.tensor(alias_inputs, dtype=torch.long),
        "seq_mask": torch.tensor(seq_mask, dtype=torch.float32),
        "seq_action_ids": torch.tensor(seq_action_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "pos_items": np.asarray(pos_items, dtype=np.int64),
        "history_sets": history_sets,
    }


class _SessionGNNNegativeSampler:
    """Mixed random/popularity negative sampler used by SessionGNN."""

    def __init__(
        self,
        n_items: int,
        popularity_probs: np.ndarray,
        n_neg: int,
        random_ratio: float,
        seed: int,
        fast_mode: bool,
        exclude_history: bool,
        unique_negatives: bool,
        oversample_factor: int,
    ) -> None:
        self.n_items = int(max(1, n_items))
        self.n_neg = int(max(1, n_neg))
        self.random_ratio = float(np.clip(random_ratio, 0.0, 1.0))
        self.fast_mode = bool(fast_mode)
        self.exclude_history = bool(exclude_history)
        self.unique_negatives = bool(unique_negatives)
        self.oversample_factor = int(max(2, oversample_factor))
        self.rng = np.random.default_rng(int(seed))

        p = np.asarray(popularity_probs, dtype=np.float64)
        if p.ndim != 1 or p.size != self.n_items:
            p = np.ones((self.n_items,), dtype=np.float64)
        p = np.clip(p, 0.0, None)
        if float(p.sum()) <= 0.0:
            p = np.ones((self.n_items,), dtype=np.float64)
        self.popularity_probs = p / float(p.sum())

    def sample_for_batch(
        self, pos_items: np.ndarray, history_sets: List[set[int]]
    ) -> np.ndarray:
        bsz = int(len(pos_items))
        neg = np.zeros((bsz, self.n_neg), dtype=np.int64)
        n_rand = int(self.n_neg * self.random_ratio)
        n_pop = self.n_neg - n_rand

        if (
            self.fast_mode
            and (not self.exclude_history)
            and (not self.unique_negatives)
            and self.n_items > 1
        ):
            if n_rand > 0:
                neg[:, :n_rand] = self.rng.integers(
                    0,
                    self.n_items,
                    size=(bsz, n_rand),
                    dtype=np.int64,
                )
            if n_pop > 0:
                neg[:, n_rand:] = self.rng.choice(
                    self.n_items,
                    size=(bsz, n_pop),
                    p=self.popularity_probs,
                ).astype(np.int64, copy=False)

            pos_col = np.asarray(pos_items, dtype=np.int64).reshape(-1, 1)
            mask = neg == pos_col
            tries = 0
            while np.any(mask) and tries < 6:
                neg[mask] = self.rng.integers(
                    0,
                    self.n_items,
                    size=int(mask.sum()),
                    dtype=np.int64,
                )
                mask = neg == pos_col
                tries += 1
            if np.any(mask):
                rows = np.nonzero(mask)[0]
                neg[mask] = (
                    np.asarray(pos_items, dtype=np.int64)[rows] + 1
                ) % self.n_items
            return neg

        all_items = np.arange(self.n_items, dtype=np.int64)
        for i, pos in enumerate(np.asarray(pos_items, dtype=np.int64)):
            banned: set[int] = set()
            if self.exclude_history and i < len(history_sets):
                banned.update(int(x) for x in history_sets[i])
            banned.add(int(pos))
            if len(banned) >= self.n_items:
                banned = {int(pos)}

            picked: List[int] = []
            picked_set: set[int] = set()
            max_attempts = self.n_neg * 200
            attempts = 0

            while len(picked) < self.n_neg and attempts < max_attempts:
                attempts += 1
                want_pop = len(picked) >= n_rand and n_pop > 0
                if want_pop:
                    c = int(self.rng.choice(self.n_items, p=self.popularity_probs))
                else:
                    c = int(self.rng.integers(0, self.n_items))
                if c in banned:
                    continue
                if self.unique_negatives and c in picked_set:
                    continue
                picked.append(c)
                picked_set.add(c)

            if len(picked) < self.n_neg:
                fill = all_items[
                    ~np.isin(all_items, np.fromiter(banned, dtype=np.int64))
                ]
                if fill.size == 0:
                    fill = np.array([int((pos + 1) % self.n_items)], dtype=np.int64)
                if self.unique_negatives:
                    fill = fill[~np.isin(fill, np.fromiter(picked_set, dtype=np.int64))]
                if fill.size == 0:
                    fill = all_items
                reps = int(np.ceil((self.n_neg - len(picked)) / max(1, fill.size)))
                filled = np.tile(fill, reps)[: self.n_neg - len(picked)].tolist()
                picked.extend(int(x) for x in filled)

            neg[i] = np.asarray(picked[: self.n_neg], dtype=np.int64)

        return neg


if nn is not None:

    class _SRGNNSequenceEncoder(nn.Module):
        """Action-aware SR-GNN sequence encoder."""

        def __init__(
            self,
            n_items: int,
            d_model: int,
            gnn_steps: int,
            dropout: float,
            max_seq_len: int,
            use_action_weighting: bool,
            action_edge_weights: Dict[int, float],
            action_attn_scale_init: float,
        ) -> None:
            super().__init__()
            self.n_items = int(n_items)
            self.d_model = int(d_model)
            self.gnn_steps = int(max(1, gnn_steps))
            self.max_seq_len = int(max(1, max_seq_len))
            self.use_action_weighting = bool(use_action_weighting)

            max_action_idx = (
                int(max(action_edge_weights.keys())) if action_edge_weights else 2
            )
            lut = torch.ones(max_action_idx + 1, dtype=torch.float32)
            lut[0] = 0.0
            for k, v in action_edge_weights.items():
                key = int(k)
                if 0 <= key <= max_action_idx:
                    lut[key] = float(v)
            self.register_buffer("action_weight_lut", lut)

            self.item_emb = nn.Embedding(self.n_items + 1, self.d_model, padding_idx=0)

            self.msg_in = nn.Linear(self.d_model, self.d_model, bias=False)
            self.msg_out = nn.Linear(self.d_model, self.d_model, bias=False)

            self.gate_r = nn.Linear(self.d_model * 3, self.d_model)
            self.gate_z = nn.Linear(self.d_model * 3, self.d_model)
            self.gate_h = nn.Linear(self.d_model * 3, self.d_model)

            self.attn_q = nn.Linear(self.d_model, self.d_model, bias=True)
            self.attn_k = nn.Linear(self.d_model, self.d_model, bias=True)
            self.attn_v = nn.Linear(self.d_model, 1, bias=False)

            self.out_proj = nn.Linear(self.d_model * 2, self.d_model, bias=False)
            self.dropout = nn.Dropout(float(dropout))
            self.attn_action_scale = nn.Parameter(
                torch.tensor(float(action_attn_scale_init), dtype=torch.float32)
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def lookup_item_vectors(
            self,
            item_idxs: torch.Tensor,
            normalize_out: bool = True,
        ) -> torch.Tensor:
            vec = self.item_emb(item_idxs + 1)  # +1 because 0 is padding
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            if normalize_out:
                vec = F.normalize(vec, dim=-1)
            return torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        def get_all_item_vectors(self, normalize_out: bool = True) -> torch.Tensor:
            ids = torch.arange(1, self.n_items + 1, device=self.item_emb.weight.device)
            vec = self.item_emb(ids)
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            if normalize_out:
                vec = F.normalize(vec, dim=-1)
            return torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        def _gnn_propagation(
            self,
            h: torch.Tensor,
            edge_src: torch.Tensor,
            edge_dst: torch.Tensor,
            edge_w: torch.Tensor,
        ) -> torch.Tensor:
            if edge_src.numel() == 0:
                return h

            for _ in range(self.gnn_steps):
                m_in = torch.zeros_like(h)
                m_out = torch.zeros_like(h)

                src_h = h[edge_src]
                dst_h = h[edge_dst]
                w = edge_w.unsqueeze(-1)

                in_msg = self.msg_in(src_h) * w
                out_msg = self.msg_out(dst_h) * w

                m_in.index_add_(0, edge_dst, in_msg)
                m_out.index_add_(0, edge_src, out_msg)

                gate_in = torch.cat([h, m_in, m_out], dim=-1)
                r = torch.sigmoid(self.gate_r(gate_in))
                z = torch.sigmoid(self.gate_z(gate_in))
                h_tilde = torch.tanh(
                    self.gate_h(torch.cat([r * h, m_in, m_out], dim=-1))
                )
                h = (1.0 - z) * h + z * h_tilde
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            return h

        def encode_batched_graph(
            self,
            node_item_ids: torch.Tensor,
            edge_src: torch.Tensor,
            edge_dst: torch.Tensor,
            edge_w: torch.Tensor,
            graph_node_ptr: torch.Tensor,
            alias_inputs: torch.Tensor,
            seq_mask: torch.Tensor,
            lengths: torch.Tensor,
            seq_action_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            del graph_node_ptr, lengths  # kept for API consistency with notebook

            h = self.item_emb(node_item_ids)
            h = self.dropout(h)
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            h = self._gnn_propagation(h, edge_src, edge_dst, edge_w)

            seq_h = h[alias_inputs]
            seq_h = torch.nan_to_num(seq_h, nan=0.0, posinf=0.0, neginf=0.0)
            local = seq_h[:, -1, :]

            q = self.attn_q(local).unsqueeze(1)
            k = self.attn_k(seq_h)
            a_logits = self.attn_v(torch.sigmoid(q + k)).squeeze(-1)

            if self.use_action_weighting and seq_action_ids is not None:
                idx = seq_action_ids.clamp(
                    min=0, max=self.action_weight_lut.numel() - 1
                )
                action_w = self.action_weight_lut[idx].to(a_logits.dtype)
                scale = 1.0 + F.softplus(self.attn_action_scale) * action_w
                a_logits = a_logits * scale

            a_logits = a_logits.masked_fill(seq_mask <= 0, -1e9)
            a = torch.softmax(a_logits, dim=-1)
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

            global_h_local = torch.sum(a.unsqueeze(-1) * seq_h, dim=1)
            session_vec = self.out_proj(torch.cat([local, global_h_local], dim=-1))
            session_vec = torch.nan_to_num(session_vec, nan=0.0, posinf=0.0, neginf=0.0)
            session_vec = F.normalize(session_vec, dim=-1)
            return torch.nan_to_num(session_vec, nan=0.0, posinf=0.0, neginf=0.0)

else:

    class _SRGNNSequenceEncoder:  # pragma: no cover - only used without torch
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "SessionGNNModel requires PyTorch to train neural embeddings."
            )


class SessionGNNModel:
    """Action-aware SR-GNN ranker adapted for exploded HRFlow interactions."""

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    ACTION_TO_IDX = {"view": 1, "apply": 2}
    DEFAULT_ACTION_EDGE_WEIGHTS = {0: 0.0, 1: 1.0, 2: 1.1920802599193465}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        # Fixed final notebook configuration (A1 action-only variant).
        return {
            "model_version": 2,
            "seed": 42,
            "device": "auto",
            "max_seq_len": 15,
            "d_model": 128,
            "gnn_steps": 1,
            "dropout": 0.37330794791414296,
            "epochs": 13,
            "batch_size": 1024,
            "num_workers": 0,
            "lr": 1e-3,
            "weight_decay": 0.00010211703461437679,
            "temperature": 0.10949463482458732,
            "n_neg": 192,
            "neg_random_ratio": 0.8005646193705664,
            "apply_view_ratio": 1.1920802599193465,
            "neg_sampler_fast_mode": True,
            "neg_sampler_exclude_history": False,
            "neg_sampler_unique": False,
            "neg_sampler_oversample": 3,
            "use_action_weighting": True,
            "action_edge_weights": dict(SessionGNNModel.DEFAULT_ACTION_EDGE_WEIGHTS),
            "action_attn_scale_init": 0.20,
            "include_target_in_fit": True,
            "smooth_alpha": 1e-6,
            "normalize_priors": True,
            "prior_as_floor": False,
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return SessionGNNModel.default_params()
        cfg = SessionGNNModel.default_params()
        cfg.update(
            {
                "apply_view_ratio": trial.suggest_float(
                    "sessiongnn_apply_view_ratio", 1.16, 1.56
                ),
                "dropout": trial.suggest_float("sessiongnn_dropout", 0.31, 0.43),
                "gnn_steps": 1,  # fixed by design
                "n_neg": trial.suggest_int("sessiongnn_n_neg", 192, 256),
                "neg_random_ratio": trial.suggest_float(
                    "sessiongnn_neg_random_ratio", 0.698, 0.843
                ),
                "temperature": trial.suggest_float(
                    "sessiongnn_temperature", 0.096, 0.114
                ),
                "weight_decay": trial.suggest_float(
                    "sessiongnn_weight_decay", 0.00003, 0.00018
                ),
            }
        )
        return cfg

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = self._harmonize_params(cfg, params)

        self.is_fitted = False
        self.job_to_idx: Dict[str, int] = {}
        self.idx_to_job: List[str] = []

        self.model: Optional[_SRGNNSequenceEncoder] = None
        self.item_matrix_cache: Optional[np.ndarray] = None

        self._prior_scores: Dict[str, float] = {}
        self.fallback_score: float = float(self.params.get("smooth_alpha", 0.0))
        self._torch_available = bool(
            torch is not None and nn is not None and F is not None
        )

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @classmethod
    def _harmonize_params(
        cls, cfg: Dict[str, Any], raw_params: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Normalize aliases and keep action-weighting params consistent."""
        out = dict(cfg)
        raw = raw_params or {}

        alias_map = {
            "apply_view_ratio_global": "apply_view_ratio",
            "dropout_global": "dropout",
            "gnn_steps_global": "gnn_steps",
            "n_neg_global": "n_neg",
            "neg_random_ratio_global": "neg_random_ratio",
            "temperature_global": "temperature",
            "weight_decay_global": "weight_decay",
        }
        for alias, canonical in alias_map.items():
            if alias in raw and canonical not in raw:
                out[canonical] = raw[alias]

        try:
            gnn_steps = int(out.get("gnn_steps", 1))
        except Exception:
            gnn_steps = 1
        out["gnn_steps"] = max(1, gnn_steps)

        ratio_default = float(cls.DEFAULT_ACTION_EDGE_WEIGHTS.get(2, 1.0))
        ratio_raw = out.get("apply_view_ratio", out.get("w_apply", ratio_default))
        try:
            ratio = float(ratio_raw)
        except Exception:
            ratio = ratio_default
        if not np.isfinite(ratio) or ratio <= 0.0:
            ratio = ratio_default
        out["apply_view_ratio"] = ratio
        out["w_apply"] = ratio

        try:
            w_view = float(out.get("w_view", 1.0))
        except Exception:
            w_view = 1.0
        if not np.isfinite(w_view) or w_view <= 0.0:
            w_view = 1.0
        out["w_view"] = w_view

        edge_weights = dict(cls.DEFAULT_ACTION_EDGE_WEIGHTS)
        raw_edges = out.get("action_edge_weights")
        if isinstance(raw_edges, dict):
            for k, v in raw_edges.items():
                try:
                    key = int(k)
                    val = float(v)
                except Exception:
                    continue
                if np.isfinite(val):
                    edge_weights[key] = val
        edge_weights[1] = float(edge_weights.get(1, 1.0))
        edge_weights[2] = ratio
        out["action_edge_weights"] = edge_weights

        return out

    @classmethod
    def _normalize_action(cls, value: Any) -> str:
        action = str(value).strip().lower()
        return "apply" if action == "apply" else "view"

    @classmethod
    def _action_to_idx(cls, action: Any) -> int:
        return int(
            cls.ACTION_TO_IDX.get(
                cls._normalize_action(action), cls.ACTION_TO_IDX["view"]
            )
        )

    @staticmethod
    def _action_weight(action: str, w_view: float, w_apply: float) -> float:
        return float(w_apply if action == "apply" else w_view)

    @classmethod
    def _normalize_interactions(cls, interactions: pd.DataFrame | None) -> pd.DataFrame:
        """Validate and normalize exploded interaction logs."""
        if interactions is None or len(interactions) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("interactions must be a pandas DataFrame or None.")

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in interactions.columns]
        if missing:
            raise ValueError(
                f"SessionGNNModel.fit() missing required interaction columns: {missing}"
            )

        out = interactions.loc[:, list(cls.REQUIRED_COLUMNS)].copy()
        out = out.dropna(subset=list(cls.REQUIRED_COLUMNS))
        out["session_id"] = out["session_id"].map(str)
        out["job_id"] = out["job_id"].map(str)
        out["action"] = out["action"].map(cls._normalize_action)
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out

    @classmethod
    def _normalize_targets(cls, targets: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize optional target rows."""
        if targets is None or len(targets) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])
        if not isinstance(targets, pd.DataFrame):
            raise TypeError("targets must be a pandas DataFrame or None.")

        required = ("session_id", "job_id")
        missing = [c for c in required if c not in targets.columns]
        if missing:
            raise ValueError(
                f"SessionGNNModel.fit() missing required target columns: {missing}"
            )

        out = targets.copy()
        out = out.dropna(subset=list(required))
        out["session_id"] = out["session_id"].map(str)
        out["job_id"] = out["job_id"].map(str)
        if "action" in out.columns:
            out["action"] = out["action"].map(cls._normalize_action)
        else:
            out["action"] = "view"
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out[["session_id", "job_id", "action", "_order"]]

    def _resolve_device(self) -> str:
        device_pref = str(self.params.get("device", "auto")).lower()
        if (not self._torch_available) or torch is None:
            return "cpu"
        if device_pref in {"cpu", "mps"}:
            if device_pref == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                return "cpu"
            return device_pref
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_vocab(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        all_job_ids: Sequence[Any] | None,
    ) -> None:
        seed_ids = (
            [str(x) for x in all_job_ids if pd.notna(x)]
            if all_job_ids is not None
            else []
        )
        known_ids = self._ordered_unique(
            seed_ids
            + interactions["job_id"].drop_duplicates().tolist()
            + (
                targets["job_id"].drop_duplicates().tolist()
                if not targets.empty
                else []
            )
        )
        self.idx_to_job = known_ids
        self.job_to_idx = {job_id: idx for idx, job_id in enumerate(known_ids)}

    def _compute_prior_scores(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
    ) -> None:
        n_items = len(self.idx_to_job)
        if n_items == 0:
            self._prior_scores = {}
            self.fallback_score = float(self.params.get("smooth_alpha", 0.0))
            return

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 1.1920802599193465))

        weights = np.zeros((n_items,), dtype=np.float64)

        if not interactions.empty:
            ordered = interactions.sort_values("_order", kind="stable")
            for _, grp in ordered.groupby("session_id", sort=False):
                # Session-level item support (first/strongest action per item).
                local: Dict[int, float] = {}
                for jid, act in zip(grp["job_id"].tolist(), grp["action"].tolist()):
                    idx = self.job_to_idx.get(str(jid))
                    if idx is None:
                        continue
                    cur = self._action_weight(str(act), w_view=w_view, w_apply=w_apply)
                    prev = local.get(int(idx), 0.0)
                    if cur > prev:
                        local[int(idx)] = float(cur)
                for idx, val in local.items():
                    weights[idx] += float(val)

        if include_target and not targets.empty:
            for jid, act in zip(targets["job_id"].tolist(), targets["action"].tolist()):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                weights[int(idx)] += self._action_weight(
                    str(act), w_view=w_view, w_apply=w_apply
                )

        smooth_alpha = float(max(0.0, self.params.get("smooth_alpha", 0.0)))
        scores = weights + smooth_alpha
        fallback = float(smooth_alpha)

        if bool(self.params.get("normalize_priors", True)):
            max_score = float(scores.max()) if scores.size > 0 else 0.0
            if max_score > 0:
                scores = scores / max_score
                fallback = fallback / max_score

        self._prior_scores = {
            job_id: float(scores[idx]) for idx, job_id in enumerate(self.idx_to_job)
        }
        self.fallback_score = float(fallback)

    def _build_session_rows(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if interactions.empty:
            return rows

        target_map: Dict[str, tuple[int, int]] = {}
        if include_target and not targets.empty:
            ordered_t = targets.sort_values("_order", kind="stable")
            for rec in ordered_t.itertuples(index=False):
                sid = str(rec.session_id)
                tgt_idx = self.job_to_idx.get(str(rec.job_id))
                if tgt_idx is None:
                    continue
                target_map[sid] = (int(tgt_idx), self._action_to_idx(rec.action))

        ordered = interactions.sort_values("_order", kind="stable")
        for sid, grp in ordered.groupby("session_id", sort=False):
            hist_item_idxs: List[int] = []
            hist_action_idxs: List[int] = []
            for jid, act in zip(grp["job_id"].tolist(), grp["action"].tolist()):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                hist_item_idxs.append(int(idx))
                hist_action_idxs.append(self._action_to_idx(act))

            if not hist_item_idxs:
                continue

            row: Dict[str, Any] = {
                "session_id": str(sid),
                "hist_item_idxs": hist_item_idxs,
                "hist_action_idxs": hist_action_idxs,
            }
            target = target_map.get(str(sid))
            if target is not None:
                row["target_item_idx"] = int(target[0])
                row["target_action_idx"] = int(target[1])
            rows.append(row)

        return rows

    def _build_training_examples(
        self,
        session_rows: List[Dict[str, Any]],
        include_target: bool,
    ) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        default_action_idx = self.ACTION_TO_IDX["view"]

        for s in session_rows:
            full_seq = list(s["hist_item_idxs"])
            full_actions = list(s["hist_action_idxs"])
            if include_target and ("target_item_idx" in s):
                full_seq.append(int(s["target_item_idx"]))
                full_actions.append(int(s.get("target_action_idx", default_action_idx)))

            # Next-item training over all prefixes.
            for t in range(1, len(full_seq)):
                examples.append(
                    {
                        "session_id": str(s["session_id"]),
                        "prefix_item_idxs": full_seq[:t],
                        "prefix_action_idxs": full_actions[:t],
                        "pos_item_idx": int(full_seq[t]),
                    }
                )

        return examples

    def _compute_sampler_popularity(
        self,
        session_rows: List[Dict[str, Any]],
        include_target: bool,
    ) -> np.ndarray:
        n_items = len(self.idx_to_job)
        if n_items == 0:
            return np.ones((1,), dtype=np.float32)

        counts = np.zeros((n_items,), dtype=np.float64)
        for row in session_rows:
            seen = set(int(x) for x in row["hist_item_idxs"])
            if include_target and ("target_item_idx" in row):
                seen.add(int(row["target_item_idx"]))
            if not seen:
                continue
            idxs = np.fromiter(seen, dtype=np.int64, count=len(seen))
            counts[idxs] += 1.0

        if float(counts.sum()) <= 0.0:
            pop = np.ones((n_items,), dtype=np.float64) / float(n_items)
        else:
            pop = counts / float(counts.sum())
        return pop.astype(np.float32, copy=False)

    def _to_device_tensor(
        self, x: Any, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)

    def _encode_from_collated(
        self, batch: Dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SessionGNNModel internal encoder is not initialized.")
        return self.model.encode_batched_graph(
            node_item_ids=self._to_device_tensor(
                batch["node_item_ids"], torch.long, device
            ),
            edge_src=self._to_device_tensor(batch["edge_src"], torch.long, device),
            edge_dst=self._to_device_tensor(batch["edge_dst"], torch.long, device),
            edge_w=self._to_device_tensor(batch["edge_w"], torch.float32, device),
            graph_node_ptr=self._to_device_tensor(
                batch["graph_node_ptr"], torch.long, device
            ),
            alias_inputs=self._to_device_tensor(
                batch["alias_inputs"], torch.long, device
            ),
            seq_mask=self._to_device_tensor(batch["seq_mask"], torch.float32, device),
            seq_action_ids=self._to_device_tensor(
                batch["seq_action_ids"], torch.long, device
            ),
            lengths=self._to_device_tensor(batch["lengths"], torch.long, device),
        )

    def _build_item_matrix_cache(self) -> None:
        if self.model is None:
            self.item_matrix_cache = None
            return
        self.model.eval()
        with torch.no_grad():
            item_vecs = (
                self.model.get_all_item_vectors(normalize_out=True).cpu().numpy()
            )
        item_vecs = np.nan_to_num(item_vecs, nan=0.0, posinf=0.0, neginf=0.0)
        self.item_matrix_cache = item_vecs.astype(np.float32, copy=False)

    def fit(self, interactions, **kwargs):
        """
        Fit SessionGNN from exploded interaction logs.

        Accepted kwargs for HRFlow compatibility:
        - targets
        - val_interactions
        - val_targets
        - job_listings
        - all_job_ids
        """
        targets = kwargs.get("targets")
        all_job_ids: Sequence[Any] | None = kwargs.get("all_job_ids")

        # Accepted for compatibility, intentionally unused.
        _ = kwargs.get("val_interactions")
        _ = kwargs.get("val_targets")
        _ = kwargs.get("job_listings")

        interactions_df = self._normalize_interactions(interactions)
        targets_df = self._normalize_targets(targets)
        include_target = bool(self.params.get("include_target_in_fit", True))

        self._build_vocab(interactions_df, targets_df, all_job_ids)
        self._compute_prior_scores(
            interactions=interactions_df,
            targets=targets_df,
            include_target=include_target,
        )

        n_items = len(self.idx_to_job)
        if n_items == 0:
            self.model = None
            self.item_matrix_cache = None
            self.is_fitted = True
            return self

        session_rows = self._build_session_rows(
            interactions=interactions_df,
            targets=targets_df,
            include_target=include_target,
        )
        examples = self._build_training_examples(
            session_rows=session_rows,
            include_target=include_target,
        )

        if len(examples) == 0:
            # No trainable prefixes (e.g., all sessions length <= 1).
            self.model = None
            self.item_matrix_cache = None
            self.is_fitted = True
            return self

        if not self._torch_available:
            # Keep popularity-smoothed fallback behavior when torch is unavailable.
            self.model = None
            self.item_matrix_cache = None
            self.is_fitted = True
            return self

        seed = int(self.params.get("seed", 42))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device(self._resolve_device())

        self.model = _SRGNNSequenceEncoder(
            n_items=n_items,
            d_model=int(self.params.get("d_model", 128)),
            gnn_steps=int(self.params.get("gnn_steps", 1)),
            dropout=float(self.params.get("dropout", 0.0)),
            max_seq_len=int(self.params.get("max_seq_len", 15)),
            use_action_weighting=bool(self.params.get("use_action_weighting", True)),
            action_edge_weights=dict(
                self.params.get("action_edge_weights", self.DEFAULT_ACTION_EDGE_WEIGHTS)
            ),
            action_attn_scale_init=float(
                self.params.get("action_attn_scale_init", 0.20)
            ),
        ).to(device)

        sampler_popularity = self._compute_sampler_popularity(
            session_rows=session_rows,
            include_target=include_target,
        )
        sampler = _SessionGNNNegativeSampler(
            n_items=n_items,
            popularity_probs=sampler_popularity,
            n_neg=int(self.params.get("n_neg", 64)),
            random_ratio=float(self.params.get("neg_random_ratio", 0.5)),
            seed=seed,
            fast_mode=bool(self.params.get("neg_sampler_fast_mode", True)),
            exclude_history=bool(self.params.get("neg_sampler_exclude_history", False)),
            unique_negatives=bool(self.params.get("neg_sampler_unique", False)),
            oversample_factor=int(self.params.get("neg_sampler_oversample", 3)),
        )

        collate_fn = partial(
            _collate_sessiongnn_examples,
            max_seq_len=int(self.params.get("max_seq_len", 15)),
            use_action_weighting=bool(self.params.get("use_action_weighting", True)),
            action_edge_weights=dict(
                self.params.get("action_edge_weights", self.DEFAULT_ACTION_EDGE_WEIGHTS)
            ),
            default_action_idx=self.ACTION_TO_IDX["view"],
        )

        batch_size = int(max(1, self.params.get("batch_size", 256)))
        num_workers = int(max(0, self.params.get("num_workers", 0)))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        loader_kwargs: Dict[str, Any] = {
            "dataset": _SessionGNNTrainingDataset(examples),
            "batch_size": min(batch_size, max(1, len(examples))),
            "shuffle": True,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "generator": generator,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        loader = DataLoader(**loader_kwargs)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params.get("lr", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        temperature = float(max(1e-6, self.params.get("temperature", 0.1)))
        epochs = int(max(1, self.params.get("epochs", 1)))

        for _ in range(epochs):
            self.model.train()
            for batch in loader:
                session_vec = self._encode_from_collated(batch, device=device)
                session_vec = torch.nan_to_num(
                    session_vec, nan=0.0, posinf=0.0, neginf=0.0
                )

                pos_items_np = np.asarray(batch["pos_items"], dtype=np.int64)
                if pos_items_np.size == 0:
                    continue

                pos_items = torch.as_tensor(
                    pos_items_np, dtype=torch.long, device=device
                )
                neg_items_np = sampler.sample_for_batch(
                    pos_items=pos_items_np,
                    history_sets=batch["history_sets"],
                )
                neg_items = torch.as_tensor(
                    neg_items_np, dtype=torch.long, device=device
                )

                pos_vec = self.model.lookup_item_vectors(pos_items, normalize_out=True)
                neg_vec = self.model.lookup_item_vectors(neg_items, normalize_out=True)

                pos_logits = (session_vec * pos_vec).sum(dim=-1, keepdim=True)
                neg_logits = torch.einsum("bd,bnd->bn", session_vec, neg_vec)

                logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        # Keep CPU weights in cache/model pickle for portability.
        self.model = self.model.to(torch.device("cpu"))
        self._build_item_matrix_cache()

        self.is_fitted = True
        return self

    def _history_to_indices(
        self, session_history: pd.DataFrame | None
    ) -> tuple[List[int], List[int]]:
        if session_history is None or len(session_history) == 0:
            return [], []
        if not isinstance(session_history, pd.DataFrame):
            raise TypeError("session_history must be a pandas DataFrame or None.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in session_history.columns]
        if missing:
            raise ValueError(
                f"SessionGNNModel.predict() missing required history columns: {missing}"
            )

        work = session_history.loc[:, list(self.REQUIRED_COLUMNS)].copy()
        work = work.dropna(subset=list(self.REQUIRED_COLUMNS))
        if work.empty:
            return [], []

        work["job_id"] = work["job_id"].map(str)
        work["action"] = work["action"].map(self._normalize_action)
        work = work.reset_index(drop=True).tail(
            int(max(1, self.params.get("max_seq_len", 15)))
        )

        item_idxs: List[int] = []
        action_idxs: List[int] = []
        for jid, action in zip(work["job_id"].tolist(), work["action"].tolist()):
            idx = self.job_to_idx.get(str(jid))
            if idx is None:
                continue
            item_idxs.append(int(idx))
            action_idxs.append(self._action_to_idx(action))
        return item_idxs, action_idxs

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """
        Score candidate jobs for one session.

        Returns a score for every input candidate id and does not remove seen items.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "SessionGNNModel must be fitted before calling predict()."
            )

        if candidate_job_ids is None:
            return {}

        candidates = [str(job_id) for job_id in candidate_job_ids]
        if not candidates:
            return {}

        out: Dict[str, float] = {
            jid: float(self._prior_scores.get(jid, self.fallback_score))
            for jid in candidates
        }

        if self.model is None or self.item_matrix_cache is None:
            return out

        hist_item_idxs, hist_action_idxs = self._history_to_indices(session_history)
        if not hist_item_idxs:
            return out

        infer_batch = _collate_sessiongnn_examples(
            batch=[
                {
                    "prefix_item_idxs": hist_item_idxs,
                    "prefix_action_idxs": hist_action_idxs,
                    "pos_item_idx": 0,
                }
            ],
            max_seq_len=int(self.params.get("max_seq_len", 15)),
            use_action_weighting=bool(self.params.get("use_action_weighting", True)),
            action_edge_weights=dict(
                self.params.get("action_edge_weights", self.DEFAULT_ACTION_EDGE_WEIGHTS)
            ),
            default_action_idx=self.ACTION_TO_IDX["view"],
        )

        self.model.eval()
        with torch.no_grad():
            session_vec = self._encode_from_collated(
                infer_batch,
                device=torch.device("cpu"),
            )
            session_vec_np = session_vec.cpu().numpy().astype(np.float32, copy=False)[0]
            session_vec_np = np.nan_to_num(
                session_vec_np, nan=0.0, posinf=0.0, neginf=0.0
            )

        known_ids: List[str] = []
        known_idxs: List[int] = []
        for jid in candidates:
            idx = self.job_to_idx.get(jid)
            if idx is None:
                continue
            known_ids.append(jid)
            known_idxs.append(int(idx))

        if not known_idxs:
            return out

        cand_mat = self.item_matrix_cache[np.asarray(known_idxs, dtype=np.int64)]
        model_scores = np.asarray(cand_mat @ session_vec_np, dtype=np.float32)
        model_scores = np.nan_to_num(
            model_scores, nan=np.nan, posinf=np.nan, neginf=np.nan
        )

        use_floor = bool(self.params.get("prior_as_floor", False))
        for jid, score in zip(known_ids, model_scores):
            if not np.isfinite(score):
                continue
            learned = float(score)
            if use_floor:
                learned = max(
                    learned, float(self._prior_scores.get(jid, self.fallback_score))
                )
            out[jid] = learned

        return out
