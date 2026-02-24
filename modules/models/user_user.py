"""Session-session collaborative filtering ranker."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class UserUserModel:
    """User-user ranker (exact IUF cosine retrieval + neighbor aggregation)."""

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    SUPPORTED_VARIANTS = {"U2U_EXACT_IUF_COSINE"}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "variant": "U2U_EXACT_IUF_COSINE",
            "iuf_exponent": 1.40,
            "neighbor_item_recency_decay": 0.94,
            "neighbor_length_norm": 0.20,
            "neighbor_position_boost": 1.32,
            "neighbor_target_boost": 1.27,
            "query_recency_decay": 0.80,
            "sim_power_gamma": 0.87,
            "top_sessions": 83,
            "w_apply": 1.59,
            "w_view": 1.53,
            "alpha": 0.5,
            "beta": 0.5,
            "bm25_b": 0.75,
            "bm25_k1": 1.5,
            "min_overlap": 1,
            "use_sequence_boost": False,
            "svd_components": 256,
            "ann_index_type": "auto",
            "ann_probe": 16,
            "ann_top_sessions": 300,
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return UserUserModel.default_params()
        prefix = "u2u"
        return {
            "variant": "U2U_EXACT_IUF_COSINE",
            "iuf_exponent": trial.suggest_float(f"{prefix}_iuf_exponent", 1.39, 1.42),
            "neighbor_item_recency_decay": trial.suggest_float(
                f"{prefix}_neighbor_item_recency_decay", 0.92, 0.95
            ),
            "neighbor_length_norm": trial.suggest_float(
                f"{prefix}_neighbor_length_norm", 0.18, 0.24
            ),
            "neighbor_position_boost": trial.suggest_float(
                f"{prefix}_neighbor_position_boost", 1.30, 1.32
            ),
            "neighbor_target_boost": trial.suggest_float(
                f"{prefix}_neighbor_target_boost", 1.27, 1.31
            ),
            "query_recency_decay": trial.suggest_float(
                f"{prefix}_query_recency_decay", 0.78, 0.80
            ),
            "sim_power_gamma": trial.suggest_float(
                f"{prefix}_sim_power_gamma", 0.86, 0.93
            ),
            "top_sessions": trial.suggest_int(f"{prefix}_top_sessions", 80, 85),
            "w_apply": trial.suggest_float(f"{prefix}_w_apply", 1.58, 1.67),
            "w_view": trial.suggest_float(f"{prefix}_w_view", 1.49, 1.54),
        }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = cfg

        self.is_fitted = False
        self.job_to_idx: Dict[str, int] = {}
        self.idx_to_job: List[str] = []

        self._iuf_vector: np.ndarray | None = None
        self._session_item_iuf_norm: sp.csr_matrix | None = None
        self._session_full_items: List[List[int]] = []
        self._session_full_actions: List[List[str]] = []

        self._prior_scores: Dict[str, float] = {}
        self.fallback_score: float = 0.0

    @staticmethod
    def _normalize_action(value: Any) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    @classmethod
    def _normalize_interactions(
        cls, interactions: pd.DataFrame | None, source_name: str
    ) -> pd.DataFrame:
        if interactions is None or len(interactions) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError(f"{source_name} must be a pandas DataFrame or None.")

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in interactions.columns]
        if missing:
            raise ValueError(f"{source_name} missing required columns: {missing}")

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
        if targets is None or len(targets) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])
        if not isinstance(targets, pd.DataFrame):
            raise TypeError("targets must be a pandas DataFrame or None.")

        if "session_id" not in targets.columns or "job_id" not in targets.columns:
            raise ValueError(
                "targets missing required columns: ['session_id', 'job_id']"
            )

        out = targets.copy()
        out = out.dropna(subset=["session_id", "job_id"])
        out["session_id"] = out["session_id"].map(str)
        out["job_id"] = out["job_id"].map(str)
        if "action" in out.columns:
            out["action"] = out["action"].map(cls._normalize_action)
        else:
            out["action"] = "view"
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out[["session_id", "job_id", "action", "_order"]]

    @staticmethod
    def _action_weight(action: str, w_view: float, w_apply: float) -> float:
        return float(w_apply if action == "apply" else w_view)

    @staticmethod
    def _truncate_history(
        items: Sequence[int], actions: Sequence[str], last_n: int
    ) -> Tuple[List[int], List[str]]:
        n = int(max(1, last_n))
        return list(items[-n:]), list(actions[-n:])

    @staticmethod
    def _topk_from_array(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = int(max(1, min(int(k), arr.size)))
        if k >= arr.size:
            idx = np.argsort(arr)[::-1]
        else:
            pool = np.argpartition(arr, -k)[-k:]
            idx = pool[np.argsort(arr[pool])[::-1]]
        vals = arr[idx]
        keep = np.isfinite(vals)
        return idx[keep].astype(np.int64), vals[keep].astype(np.float32)

    def _build_query_weight_vector(
        self, hist_items: List[int], hist_actions: List[str]
    ) -> sp.csr_matrix:
        n_items = len(self.idx_to_job)
        last_n = int(self.params.get("last_n", 8))
        items, actions = self._truncate_history(hist_items, hist_actions, last_n)
        n = len(items)
        if n == 0 or n_items == 0:
            return sp.csr_matrix((1, n_items), dtype=np.float32)

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        q_decay = float(self.params.get("query_recency_decay", 0.85))
        acc: Dict[int, float] = defaultdict(float)

        for pos, (it, act) in enumerate(zip(items, actions)):
            w = self._action_weight(act, w_view, w_apply) * (q_decay ** (n - 1 - pos))
            acc[int(it)] += float(w)

        if not acc:
            return sp.csr_matrix((1, n_items), dtype=np.float32)

        cols = np.fromiter(acc.keys(), dtype=np.int64)
        vals = np.fromiter(acc.values(), dtype=np.float32)
        rows = np.zeros_like(cols, dtype=np.int64)
        return sp.csr_matrix((vals, (rows, cols)), shape=(1, n_items), dtype=np.float32)

    def _retrieve_neighbors(
        self, hist_items: List[int], hist_actions: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._session_item_iuf_norm is None or self._iuf_vector is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        query_weighted = self._build_query_weight_vector(hist_items, hist_actions)
        if query_weighted.nnz == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q_iuf = query_weighted.multiply(self._iuf_vector.reshape(1, -1))
        q_norm = normalize(q_iuf, norm="l2", axis=1)
        if q_norm.nnz == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        sims = (
            (q_norm @ self._session_item_iuf_norm.T)
            .toarray()
            .ravel()
            .astype(np.float32)
        )
        top_sessions = int(self.params.get("top_sessions", 100))
        idx, vals = self._topk_from_array(sims, top_sessions)
        keep = vals > 0
        return idx[keep], vals[keep]

    def _aggregate_neighbor_item_scores(
        self, neighbor_ids: np.ndarray, neighbor_sims: np.ndarray
    ) -> np.ndarray:
        n_items = len(self.idx_to_job)
        scores = np.zeros(n_items, dtype=np.float64)

        gamma = float(self.params.get("sim_power_gamma", 1.0))
        n_decay = float(self.params.get("neighbor_item_recency_decay", 0.9))
        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        use_sequence_boost = bool(self.params.get("use_sequence_boost", False))
        pos_boost = float(self.params.get("neighbor_position_boost", 1.0))
        target_boost = float(self.params.get("neighbor_target_boost", 1.0))

        for sid, sim in zip(neighbor_ids, neighbor_sims):
            if not np.isfinite(sim) or sim <= 0:
                continue
            sid_i = int(sid)
            if sid_i < 0 or sid_i >= len(self._session_full_items):
                continue

            items = self._session_full_items[sid_i]
            acts = self._session_full_actions[sid_i]
            n = len(items)
            if n == 0:
                continue

            base = float(sim) ** gamma
            for pos, (it, act) in enumerate(zip(items, acts)):
                aw = self._action_weight(act, w_view, w_apply)
                rw = n_decay ** (n - 1 - pos)
                boost = 1.0
                if use_sequence_boost:
                    if pos >= max(0, n - 2):
                        boost *= pos_boost
                    if pos == n - 1:
                        boost *= target_boost
                scores[int(it)] += base * aw * rw * boost

        return scores.astype(np.float32)

    def _build_prior_scores(
        self, interactions_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> None:
        if not self.idx_to_job:
            self._prior_scores = {}
            self.fallback_score = 0.0
            return

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        alpha = 1.0
        counts = {jid: 0.0 for jid in self.idx_to_job}

        for df in (interactions_df, targets_df):
            if df.empty:
                continue
            for jid, act in zip(df["job_id"].tolist(), df["action"].tolist()):
                if jid not in counts:
                    continue
                counts[jid] += self._action_weight(str(act), w_view, w_apply)

        total = float(sum(counts.values()))
        n_items = max(len(self.idx_to_job), 1)
        denom = total + alpha * n_items

        self._prior_scores = {
            jid: float((raw + alpha) / denom) for jid, raw in counts.items()
        }
        self.fallback_score = float(alpha / denom)

    def fit(self, interactions, **kwargs):
        """
        Fit the user-user artifact from exploded interaction logs.

        Accepted kwargs for HRFlow compatibility:
        `targets`, `val_interactions`, `val_targets`, `job_listings`, `all_job_ids`.
        """
        targets = kwargs.get("targets")
        _ = kwargs.get("val_interactions")
        _ = kwargs.get("val_targets")
        _ = kwargs.get("job_listings")
        all_job_ids = kwargs.get("all_job_ids")

        variant = str(self.params.get("variant", "U2U_EXACT_IUF_COSINE")).upper()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported user-user variant: {variant}")

        interactions_df = self._normalize_interactions(interactions, "interactions")
        targets_df = self._normalize_targets(targets)

        extra_ids = (
            [str(x) for x in all_job_ids if pd.notna(x)]
            if all_job_ids is not None
            else []
        )
        known_ids = self._ordered_unique(
            extra_ids
            + interactions_df["job_id"].drop_duplicates().tolist()
            + targets_df["job_id"].drop_duplicates().tolist()
        )
        self.idx_to_job = list(known_ids)
        self.job_to_idx = {jid: idx for idx, jid in enumerate(self.idx_to_job)}

        n_items = len(self.idx_to_job)
        if n_items == 0:
            self._iuf_vector = np.zeros((0,), dtype=np.float32)
            self._session_item_iuf_norm = sp.csr_matrix((0, 0), dtype=np.float32)
            self._session_full_items = []
            self._session_full_actions = []
            self._build_prior_scores(interactions_df, targets_df)
            self.is_fitted = True
            return self

        target_by_session: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        if not targets_df.empty:
            for sid, jid, act in zip(
                targets_df["session_id"].tolist(),
                targets_df["job_id"].tolist(),
                targets_df["action"].tolist(),
            ):
                target_by_session[str(sid)].append((str(jid), str(act)))

        last_n = int(self.params.get("last_n", 8))
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        self._session_full_items = []
        self._session_full_actions = []

        grouped = interactions_df.groupby("session_id", sort=False)
        for sid, grp in grouped:
            grp = grp.sort_values("_order", kind="stable")
            hist_items: List[int] = []
            hist_actions: List[str] = []

            for jid, act in zip(grp["job_id"].tolist(), grp["action"].tolist()):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                hist_items.append(int(idx))
                hist_actions.append(str(act))

            hist_items, hist_actions = self._truncate_history(
                hist_items, hist_actions, last_n
            )
            if not hist_items:
                continue

            session_idx = len(self._session_full_items)
            for it in set(hist_items):
                rows.append(session_idx)
                cols.append(int(it))
                vals.append(1.0)

            full_items = list(hist_items)
            full_actions = list(hist_actions)
            for tjid, tact in target_by_session.get(str(sid), []):
                tidx = self.job_to_idx.get(str(tjid))
                if tidx is None:
                    continue
                full_items.append(int(tidx))
                full_actions.append(str(tact))

            self._session_full_items.append(full_items)
            self._session_full_actions.append(full_actions)

        binary = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(len(self._session_full_items), n_items),
            dtype=np.float32,
        ).tocsr()

        n_sessions = max(1, binary.shape[0])
        df = binary.getnnz(axis=0).astype(np.float64)
        iuf_exp = float(self.params.get("iuf_exponent", 1.0))
        self._iuf_vector = np.power(
            np.log((1.0 + n_sessions) / (1.0 + df) + 1.0), iuf_exp
        ).astype(np.float32)

        mat_iuf = binary.multiply(self._iuf_vector.reshape(1, -1)).tocsr()
        self._session_item_iuf_norm = normalize(mat_iuf, norm="l2", axis=1).tocsr()

        self._build_prior_scores(interactions_df, targets_df)
        self.is_fitted = True
        return self

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """
        Score candidates for one session using user-user neighbors.

        Returns a score for every candidate id passed in. Seen-item filtering is
        intentionally not applied here (hybrid layer handles filtering/ranking).
        """
        if not self.is_fitted:
            raise RuntimeError("UserUserModel must be fitted before calling predict().")
        if candidate_job_ids is None:
            return {}

        hist_df = self._normalize_interactions(session_history, "session_history")
        hist_items: List[int] = []
        hist_actions: List[str] = []
        if not hist_df.empty:
            for jid, act in zip(hist_df["job_id"].tolist(), hist_df["action"].tolist()):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                hist_items.append(int(idx))
                hist_actions.append(str(act))

        neighbor_ids, neighbor_sims = self._retrieve_neighbors(hist_items, hist_actions)
        signal_scores = (
            self._aggregate_neighbor_item_scores(neighbor_ids, neighbor_sims)
            if len(neighbor_ids) > 0
            else np.zeros(len(self.idx_to_job), dtype=np.float32)
        )

        max_signal = float(np.max(signal_scores)) if signal_scores.size > 0 else 0.0
        if max_signal > 0:
            signal_scores = signal_scores / max_signal

        out: Dict[str, float] = {}
        for candidate in candidate_job_ids:
            jid = str(candidate)
            idx = self.job_to_idx.get(jid)

            if idx is not None and idx < signal_scores.size:
                score = float(signal_scores[idx])
                if score > 0 and np.isfinite(score):
                    out[jid] = score
                    continue

            out[jid] = float(self._prior_scores.get(jid, self.fallback_score))

        return out
