"""Item-item collaborative filtering ranker."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


class ItemItemModel:
    """
    Item-item collaborative ranker using session-weighted interaction signals.

    The implementation follows the POC #4 "I-I" logic:
    1. Build a weighted session-item matrix.
    2. Compute item-item similarity (Tversky or Cosine).
    3. Keep top-K neighbors per item.
    4. Score candidates with a weighted query vector from session history.
    """

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    SUPPORTED_VARIANTS = {"II_TVERSKY", "II_COSINE"}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "variant": "II_TVERSKY",
            "last_n": 10,
            "neighborhood_size": 100,
            "query_recency_decay": 0.74,
            "shrinkage": 17,
            "train_recency_decay": 0.89,
            "tversky_alpha": 0.29,
            "tversky_beta": 0.03,
            "w_apply": 1.63,
            "w_view": 1.24,
            "include_self_transition": False,
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return ItemItemModel.default_params()
        prefix = "item_item"
        return {
            "variant": "II_TVERSKY",
            "last_n": trial.suggest_int(f"{prefix}_last_n", 9, 12),
            "neighborhood_size": 100,
            "query_recency_decay": trial.suggest_float(
                f"{prefix}_query_recency_decay", 0.73, 0.76
            ),
            "shrinkage": trial.suggest_int(f"{prefix}_shrinkage", 16, 17),
            "train_recency_decay": trial.suggest_float(
                f"{prefix}_train_recency_decay", 0.88, 0.90
            ),
            "tversky_alpha": trial.suggest_float(f"{prefix}_tversky_alpha", 0.21, 0.30),
            "tversky_beta": trial.suggest_float(f"{prefix}_tversky_beta", 0.02, 0.04),
            "w_apply": trial.suggest_float(f"{prefix}_w_apply", 1.50, 1.72),
            "w_view": trial.suggest_float(f"{prefix}_w_view", 1.23, 1.29),
        }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = cfg
        self.is_fitted = False

        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: List[str] = []
        self.similarity_matrix: sp.csr_matrix | None = None
        self.item_prior_scores: np.ndarray = np.zeros((0,), dtype=np.float32)
        self.fallback_score: float = 0.0

    @staticmethod
    def _normalize_action(value: Any) -> str:
        return str(value).strip().lower()

    @classmethod
    def _normalize_interactions(cls, interactions: pd.DataFrame | None) -> pd.DataFrame:
        """Validate and normalize exploded interaction logs."""
        if interactions is None or len(interactions) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in interactions.columns]
        if missing:
            raise ValueError(
                f"ItemItemModel.fit() missing required interaction columns: {missing}"
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
        """Normalize target rows (optional in fit)."""
        if targets is None or len(targets) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])

        required = ("session_id", "job_id")
        missing = [c for c in required if c not in targets.columns]
        if missing:
            raise ValueError(
                f"ItemItemModel.fit() missing required target columns: {missing}"
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

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _action_weight(action: str, w_view: float, w_apply: float) -> float:
        return float(w_apply if action == "apply" else w_view)

    def _build_vocab(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        all_job_ids: Sequence[Any] | None,
    ) -> None:
        """Create a stable item vocabulary for train/inference."""
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
        self.idx_to_item = known_ids
        self.item_to_idx = {job_id: idx for idx, job_id in enumerate(known_ids)}

    def _build_session_item_matrix(
        self, interactions: pd.DataFrame, targets: pd.DataFrame
    ) -> sp.csc_matrix:
        """
        Build weighted session-item matrix X (sessions x items), mirroring POC #4.

        Sessions are weighted by action type and recency within each session.
        If targets are provided, they are added as an extra positive boost.
        """
        n_items = len(self.idx_to_item)
        if n_items == 0:
            return sp.csc_matrix((0, 0), dtype=np.float32)

        session_ids = self._ordered_unique(
            interactions["session_id"].drop_duplicates().tolist()
            + (
                targets["session_id"].drop_duplicates().tolist()
                if not targets.empty
                else []
            )
        )
        if not session_ids:
            return sp.csc_matrix((0, n_items), dtype=np.float32)

        session_items: Dict[str, List[int]] = defaultdict(list)
        session_actions: Dict[str, List[str]] = defaultdict(list)
        for row in interactions.itertuples(index=False):
            idx = self.item_to_idx.get(row.job_id)
            if idx is None:
                continue
            session_items[row.session_id].append(int(idx))
            session_actions[row.session_id].append(str(row.action))

        target_items: Dict[str, List[int]] = defaultdict(list)
        for row in targets.itertuples(index=False):
            idx = self.item_to_idx.get(row.job_id)
            if idx is None:
                continue
            target_items[row.session_id].append(int(idx))

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        decay = float(self.params.get("train_recency_decay", 0.9))
        target_boost = max(w_apply, 1.0)

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for sidx, sid in enumerate(session_ids):
            local_w: Dict[int, float] = defaultdict(float)
            items = session_items.get(sid, [])
            actions = session_actions.get(sid, [])
            L = len(items)

            for pos, (it, act) in enumerate(zip(items, actions)):
                w = self._action_weight(act, w_view, w_apply) * (decay ** (L - 1 - pos))
                local_w[int(it)] += float(w)

            for tgt in target_items.get(sid, []):
                local_w[int(tgt)] += float(target_boost)

            for iid, weight in local_w.items():
                rows.append(int(sidx))
                cols.append(int(iid))
                vals.append(float(weight))

        matrix = sp.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(session_ids), n_items),
            dtype=np.float32,
        )
        return matrix.tocsc()

    @staticmethod
    def _sparsify_topk_rows(sim: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """Keep only the top-k neighbors per row."""
        sim = sim.tocsr()
        if sim.shape[0] == 0:
            return sim
        if k <= 0:
            return sp.csr_matrix(sim.shape, dtype=np.float32)
        if k >= sim.shape[1]:
            return sim

        data = sim.data
        indices = sim.indices
        indptr = sim.indptr

        out_data: List[float] = []
        out_indices: List[int] = []
        out_indptr = [0]

        for i in range(sim.shape[0]):
            st, en = indptr[i], indptr[i + 1]
            if st == en:
                out_indptr.append(len(out_data))
                continue

            row_data = data[st:en]
            row_idx = indices[st:en]

            if row_data.size > k:
                keep = np.argpartition(row_data, -k)[-k:]
                row_data = row_data[keep]
                row_idx = row_idx[keep]

            # Deterministic order for pickling/reproducibility.
            order = np.lexsort((row_idx, -row_data))
            row_data = row_data[order]
            row_idx = row_idx[order]

            out_data.extend(row_data.tolist())
            out_indices.extend(row_idx.tolist())
            out_indptr.append(len(out_data))

        return sp.csr_matrix(
            (np.asarray(out_data, dtype=np.float32), out_indices, out_indptr),
            shape=sim.shape,
            dtype=np.float32,
        )

    def _build_similarity_matrix(self, x_csc: sp.csc_matrix) -> sp.csr_matrix:
        """Build item-item similarity matrix from weighted session-item matrix."""
        variant = str(self.params.get("variant", "II_TVERSKY")).upper()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unknown item-item variant: {variant}")

        n_items = x_csc.shape[1]
        if n_items == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        shrinkage = max(float(self.params.get("shrinkage", 0.0)), 0.0)
        support_proxy: sp.csr_matrix | None = None

        if variant == "II_COSINE":
            x_items_raw = x_csc.T.tocsr().astype(np.float32)
            if shrinkage > 0:
                support_proxy = (x_items_raw @ x_items_raw.T).tocsr().astype(np.float32)

            norms = np.asarray(
                np.sqrt(x_items_raw.multiply(x_items_raw).sum(axis=1))
            ).ravel()
            norms = np.where(norms > 0.0, norms, 1.0).astype(np.float32)

            x_items = x_items_raw.copy()
            data = x_items.data
            indptr = x_items.indptr
            for i in range(n_items):
                st, en = indptr[i], indptr[i + 1]
                if st < en:
                    data[st:en] /= norms[i]

            sim = (x_items @ x_items.T).tocsr().astype(np.float32)

            if shrinkage > 0 and support_proxy is not None and support_proxy.nnz > 0:
                # sim *= support / (support + shrinkage)
                if (
                    sim.nnz == support_proxy.nnz
                    and np.array_equal(sim.indptr, support_proxy.indptr)
                    and np.array_equal(sim.indices, support_proxy.indices)
                ):
                    sup = support_proxy.data.astype(np.float32, copy=False)
                    sim.data *= sup / (sup + np.float32(shrinkage))
                else:
                    sup = support_proxy.tocoo()
                    factor = sp.csr_matrix(
                        (
                            sup.data / (sup.data + np.float32(shrinkage)),
                            (sup.row, sup.col),
                        ),
                        shape=support_proxy.shape,
                        dtype=np.float32,
                    )
                    sim = sim.multiply(factor).tocsr()

        else:  # II_TVERSKY
            alpha = float(self.params.get("tversky_alpha", 1.0))
            beta = float(self.params.get("tversky_beta", 1.0))

            item_pops = np.asarray(x_csc.sum(axis=0)).ravel().astype(np.float64)
            x_items = x_csc.T.tocsr().astype(np.float32)
            inter = (x_items @ x_items.T).tocoo()

            if inter.nnz == 0:
                sim = sp.csr_matrix((n_items, n_items), dtype=np.float32)
            else:
                rows = inter.row.astype(np.int64, copy=False)
                cols = inter.col.astype(np.int64, copy=False)
                inter_data = inter.data.astype(np.float64, copy=False)

                ri = item_pops[rows]
                rj = item_pops[cols]
                denom = inter_data * (1.0 - alpha - beta) + alpha * ri + beta * rj
                denom = np.where(denom > 1e-12, denom, 1.0)

                sim_data = (inter_data / denom).astype(np.float32)
                if shrinkage > 0:
                    support = inter_data.astype(np.float32)
                    sim_data *= support / (support + np.float32(shrinkage))

                sim = sp.csr_matrix(
                    (sim_data, (rows, cols)),
                    shape=(n_items, n_items),
                    dtype=np.float32,
                )

        k_neighbors = int(self.params.get("neighborhood_size", 100))
        sim = self._sparsify_topk_rows(sim, k_neighbors)

        if not bool(self.params.get("include_self_transition", False)):
            sim.setdiag(0.0)
            sim.eliminate_zeros()

        return sim.tocsr()

    def _compute_item_priors(self, x_csc: sp.csc_matrix) -> None:
        """
        Build smoothed popularity priors used as fallbacks.

        Priors are intentionally small positive scores to avoid returning hard zeros
        for unknown/unsupported candidate items.
        """
        n_items = len(self.idx_to_item)
        if n_items == 0:
            self.item_prior_scores = np.zeros((0,), dtype=np.float32)
            self.fallback_score = 0.0
            return

        raw = np.asarray(x_csc.sum(axis=0)).ravel().astype(np.float64)
        smooth = max(1.0, float(self.params.get("shrinkage", 0.0)))
        denom = float(raw.sum() + smooth * n_items)

        if denom <= 0:
            value = float(1.0 / n_items)
            self.item_prior_scores = np.full((n_items,), value, dtype=np.float32)
            self.fallback_score = value
            return

        priors = (raw + smooth) / denom
        self.item_prior_scores = priors.astype(np.float32, copy=False)
        self.fallback_score = float(smooth / denom)

    def _build_query_vector(
        self, session_history: pd.DataFrame | None
    ) -> sp.csr_matrix:
        """Convert a single-session history slice to a weighted sparse query."""
        n_items = len(self.idx_to_item)
        if n_items == 0 or session_history is None or len(session_history) == 0:
            return sp.csr_matrix((1, n_items), dtype=np.float32)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in session_history.columns]
        if missing:
            raise ValueError(
                f"ItemItemModel.predict() missing required history columns: {missing}"
            )

        work = session_history.loc[:, ["job_id", "action"]].copy()
        work = work.dropna(subset=["job_id", "action"])
        if work.empty:
            return sp.csr_matrix((1, n_items), dtype=np.float32)

        work["job_id"] = work["job_id"].map(str)
        work["action"] = work["action"].map(self._normalize_action)

        last_n = int(max(1, self.params.get("last_n", 10)))
        work = work.tail(last_n)
        items = work["job_id"].tolist()
        actions = work["action"].tolist()
        L = len(items)

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        decay = float(self.params.get("query_recency_decay", 0.9))

        acc: Dict[int, float] = defaultdict(float)
        for pos, (job_id, action) in enumerate(zip(items, actions)):
            idx = self.item_to_idx.get(job_id)
            if idx is None:
                continue
            w = self._action_weight(action, w_view, w_apply) * (decay ** (L - 1 - pos))
            acc[int(idx)] += float(w)

        if not acc:
            return sp.csr_matrix((1, n_items), dtype=np.float32)

        cols = np.fromiter(acc.keys(), dtype=np.int64)
        vals = np.fromiter(acc.values(), dtype=np.float32)
        rows = np.zeros(cols.shape[0], dtype=np.int64)
        return sp.csr_matrix((vals, (rows, cols)), shape=(1, n_items), dtype=np.float32)

    def fit(self, interactions, **kwargs):
        """
        Fit the item-item model from exploded interaction logs.

        Accepted kwargs for HRFlow compatibility:
        - targets
        - val_interactions
        - val_targets
        - job_listings
        - all_job_ids
        """
        targets = kwargs.get("targets")
        all_job_ids: Sequence[Any] | None = kwargs.get("all_job_ids")

        # Accepted but unused for this model; kept for pipeline compatibility.
        _ = kwargs.get("val_interactions")
        _ = kwargs.get("val_targets")
        _ = kwargs.get("job_listings")

        interactions_df = self._normalize_interactions(interactions)
        targets_df = self._normalize_targets(targets)

        self._build_vocab(interactions_df, targets_df, all_job_ids)
        x_csc = self._build_session_item_matrix(interactions_df, targets_df)
        self.similarity_matrix = self._build_similarity_matrix(x_csc)
        self._compute_item_priors(x_csc)

        self.is_fitted = True
        return self

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """
        Score all candidate jobs for one session.

        Returns a score for every input candidate id. Seen-item filtering is not
        applied here (handled by the hybrid ranking layer).
        """
        if not self.is_fitted or self.similarity_matrix is None:
            raise RuntimeError("ItemItemModel must be fitted before calling predict().")

        if candidate_job_ids is None:
            return {}

        candidates = [str(job_id) for job_id in candidate_job_ids]
        if not candidates:
            return {}

        query = self._build_query_vector(session_history)
        personalized_scores: np.ndarray | None = None
        if query.nnz > 0 and self.similarity_matrix.shape[0] > 0:
            personalized_scores = np.asarray(
                (query @ self.similarity_matrix).toarray()
            ).ravel()

        scores: Dict[str, float] = {}
        n_priors = int(self.item_prior_scores.shape[0])
        for job_id in candidates:
            idx = self.item_to_idx.get(job_id)
            if idx is None:
                scores[job_id] = float(self.fallback_score)
                continue

            prior = (
                float(self.item_prior_scores[idx])
                if idx < n_priors
                else float(self.fallback_score)
            )
            if personalized_scores is None:
                scores[job_id] = prior
                continue

            value = float(personalized_scores[idx])
            if np.isfinite(value) and value > 0.0:
                scores[job_id] = value
            else:
                scores[job_id] = max(prior, float(self.fallback_score))

        return scores
