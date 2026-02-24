"""Matrix factorization implicit-based ranker."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

try:
    import implicit as implicit_lib
except Exception:  # pragma: no cover - optional dependency
    implicit_lib = None


class MFModel:
    """Session-as-user MF model with cold-user fold-in inference."""

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    SUPPORTED_VARIANTS = {"MF_IMPLICIT_BPR", "MF_IMPLICIT_ALS", "MF_IMPLICIT_LMF"}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "variant": "MF_IMPLICIT_BPR",
            "last_item_boost": 1.28,
            "last_n": 7,
            "lr_all": 0.0136,
            "n_epochs": 74,
            "n_factors": 64,
            "query_recency_decay": 0.86,
            "rating_clip_max": 2.47,
            "reg_all": 0.0141,
            "sim_power_gamma": 1.91,
            "train_recency_decay": 0.95,
            "w_apply": 1.48,
            "w_view": 0.76,
            "foldin_mode": "last_item_boost",
            "foldin_source": "qi",
            "include_target_in_fit": True,
            "rating_transform": "log1p",
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return MFModel.default_params()
        prefix = "mf"
        return {
            "variant": "MF_IMPLICIT_BPR",
            "last_item_boost": trial.suggest_float(
                f"{prefix}_last_item_boost", 1.27, 1.35
            ),
            "last_n": 7,
            "lr_all": trial.suggest_float(f"{prefix}_lr_all", 0.0132, 0.0140),
            "n_epochs": trial.suggest_int(f"{prefix}_n_epochs", 74, 85),
            "n_factors": trial.suggest_categorical(
                f"{prefix}_n_factors", [64, 128, 256]
            ),
            "query_recency_decay": trial.suggest_float(
                f"{prefix}_query_recency_decay", 0.86, 0.90
            ),
            "rating_clip_max": trial.suggest_float(
                f"{prefix}_rating_clip_max", 2.00, 2.48
            ),
            "reg_all": trial.suggest_float(f"{prefix}_reg_all", 0.0141, 0.0215),
            "sim_power_gamma": trial.suggest_float(
                f"{prefix}_sim_power_gamma", 1.78, 1.91
            ),
            "train_recency_decay": trial.suggest_float(
                f"{prefix}_train_recency_decay", 0.78, 0.99
            ),
            "w_apply": trial.suggest_float(f"{prefix}_w_apply", 1.48, 1.73),
            "w_view": trial.suggest_float(f"{prefix}_w_view", 0.70, 0.96),
        }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = cfg

        self.is_fitted = False
        self.job_to_idx: Dict[str, int] = {}
        self.idx_to_job: List[str] = []

        self._item_factors = np.zeros((0, 0), dtype=np.float32)
        self._item_factors_unit = np.zeros((0, 0), dtype=np.float32)
        self._item_bias = np.zeros((0,), dtype=np.float32)
        self._global_bias = 0.0
        self._train_item_support = np.zeros((0,), dtype=np.float32)
        self._item_has_signal = np.zeros((0,), dtype=bool)
        self._backend_name = "unfitted"

        self._prior_scores: Dict[str, float] = {}
        self.fallback_score: float = 0.0

    @staticmethod
    def _normalize_action(value: Any) -> str:
        return str(value).strip().lower()

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

    @staticmethod
    def _history_slice(
        items: Sequence[int], actions: Sequence[str], last_n: int
    ) -> Tuple[List[int], List[str]]:
        n = int(max(1, last_n))
        return list(items[-n:]), list(actions[-n:])

    @staticmethod
    def _action_weight(action: str, w_view: float, w_apply: float) -> float:
        return float(w_apply if action == "apply" else w_view)

    @staticmethod
    def _transform_rating(value: float, transform: str) -> float:
        z = float(max(0.0, value))
        if transform == "log1p":
            return float(np.log1p(z))
        if transform == "sqrt":
            return float(np.sqrt(z))
        return z

    @staticmethod
    def _normalize_rows(mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return arr.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (arr / norms).astype(np.float32)

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

        missing = [c for c in ("session_id", "job_id") if c not in targets.columns]
        if missing:
            raise ValueError(f"targets missing required columns: {missing}")

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

    def _build_job_vocab(
        self,
        interactions_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        all_job_ids: Sequence[Any] | None,
    ) -> None:
        ordered: List[str] = []
        if not interactions_df.empty:
            ordered.extend(interactions_df["job_id"].tolist())
        if not targets_df.empty:
            ordered.extend(targets_df["job_id"].tolist())
        self.idx_to_job = self._ordered_unique(ordered)
        self.job_to_idx = {job_id: idx for idx, job_id in enumerate(self.idx_to_job)}

        self._all_job_universe = set(self.idx_to_job)
        if all_job_ids is not None:
            self._all_job_universe.update(str(x) for x in all_job_ids if pd.notna(x))

    def _build_weighted_interaction_table(
        self, interactions_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """Build weighted user-item table from exploded logs (POC #3 format)."""
        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        decay = float(self.params.get("train_recency_decay", 0.9))
        last_n = int(self.params.get("last_n", 8))
        include_target = bool(self.params.get("include_target_in_fit", True))
        transform = str(self.params.get("rating_transform", "log1p"))
        clip_max = float(self.params.get("rating_clip_max", 5.0))

        histories: Dict[str, Tuple[List[int], List[str]]] = {}
        session_order: List[str] = []
        if not interactions_df.empty:
            ordered_hist = interactions_df.sort_values("_order", kind="stable")
            for sid, grp in ordered_hist.groupby("session_id", sort=False):
                items: List[int] = []
                actions: List[str] = []
                for jid, act in zip(grp["job_id"].tolist(), grp["action"].tolist()):
                    idx = self.job_to_idx.get(jid, -1)
                    if idx < 0:
                        continue
                    items.append(int(idx))
                    actions.append(str(act))
                histories[str(sid)] = (items, actions)
                session_order.append(str(sid))

        if not targets_df.empty:
            for sid in targets_df.sort_values("_order", kind="stable")[
                "session_id"
            ].tolist():
                sid_s = str(sid)
                if sid_s not in histories:
                    histories[sid_s] = ([], [])
                    session_order.append(sid_s)

        if not session_order:
            return pd.DataFrame(columns=["uid_int", "iid_int", "rating"]), 0

        sid_to_uid = {sid: uid for uid, sid in enumerate(session_order)}

        target_items_by_sid: Dict[str, List[int]] = defaultdict(list)
        if include_target and not targets_df.empty:
            ordered_targets = targets_df.sort_values("_order", kind="stable")
            for sid, jid in zip(
                ordered_targets["session_id"].tolist(),
                ordered_targets["job_id"].tolist(),
            ):
                tgt_idx = self.job_to_idx.get(str(jid), -1)
                if tgt_idx >= 0:
                    target_items_by_sid[str(sid)].append(int(tgt_idx))

        acc: Dict[Tuple[int, int], float] = defaultdict(float)
        event_count: Dict[Tuple[int, int], int] = defaultdict(int)
        target_boost = max(w_apply, 1.0)

        for sid in session_order:
            uid = int(sid_to_uid[sid])
            hist_items, hist_actions = histories.get(sid, ([], []))
            hist_items, hist_actions = self._history_slice(
                hist_items, hist_actions, last_n
            )
            length = len(hist_items)

            for pos, (iid, act) in enumerate(zip(hist_items, hist_actions)):
                w = self._action_weight(act, w_view, w_apply) * (
                    decay ** (length - 1 - pos)
                )
                key = (uid, int(iid))
                acc[key] += float(w)
                event_count[key] += 1

            if include_target:
                for tgt in target_items_by_sid.get(sid, []):
                    key_t = (uid, int(tgt))
                    acc[key_t] += float(target_boost)
                    event_count[key_t] += 1

        rows = []
        for (uid, iid), raw_signal in acc.items():
            rating = self._transform_rating(raw_signal, transform)
            rating = float(np.clip(rating, 1e-6, clip_max))
            rows.append(
                {
                    "uid_int": int(uid),
                    "iid_int": int(iid),
                    "rating": rating,
                    "raw_signal": float(raw_signal),
                    "events": int(event_count[(uid, iid)]),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["uid_int", "iid_int", "rating"]), len(
                session_order
            )
        return pd.DataFrame(rows), len(session_order)

    @staticmethod
    def _user_item_matrix(
        interactions_df: pd.DataFrame, n_users: int, n_items: int
    ) -> sp.csr_matrix:
        if interactions_df.empty:
            return sp.csr_matrix((n_users, n_items), dtype=np.float32)
        rows = interactions_df["uid_int"].to_numpy(dtype=np.int64)
        cols = interactions_df["iid_int"].to_numpy(dtype=np.int64)
        vals = interactions_df["rating"].to_numpy(dtype=np.float32)
        return sp.coo_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        ).tocsr()

    @staticmethod
    def _build_item_support(interactions_df: pd.DataFrame, n_items: int) -> np.ndarray:
        support = np.zeros((n_items,), dtype=np.float32)
        if interactions_df.empty:
            return support
        sums = interactions_df.groupby("iid_int", sort=False)["rating"].sum()
        for iid, val in sums.items():
            idx = int(iid)
            if 0 <= idx < n_items:
                support[idx] = float(val)
        return support

    def _build_smoothed_priors(self, support: np.ndarray) -> None:
        alpha = 1.0
        universe_size = max(
            len(getattr(self, "_all_job_universe", set())), len(support), 1
        )
        total = float(np.sum(np.maximum(support, 0.0)))
        denom = total + alpha * float(universe_size + 1)
        if denom <= 0:
            self._prior_scores = {}
            self.fallback_score = 0.0
            return

        self._prior_scores = {}
        for idx, job_id in enumerate(self.idx_to_job):
            raw = float(support[idx]) if idx < len(support) else 0.0
            self._prior_scores[job_id] = float((raw + alpha) / denom)
        self.fallback_score = float(alpha / denom)

    def _train_implicit_item_factors(
        self, user_item: sp.csr_matrix, variant: str
    ) -> np.ndarray:
        if implicit_lib is None:
            raise RuntimeError(
                "implicit library is not available. Install 'implicit' to use implicit MF."
            )

        n_factors = int(max(1, self.params.get("n_factors", 64)))
        n_epochs = int(max(1, self.params.get("n_epochs", 50)))
        lr_all = float(self.params.get("lr_all", 0.01))
        reg_all = float(self.params.get("reg_all", 0.02))

        model = None
        num_threads = 1  # keep deterministic behavior
        seed = 42

        if variant == "MF_IMPLICIT_ALS":
            kwargs = {
                "factors": n_factors,
                "regularization": reg_all,
                "iterations": n_epochs,
                "random_state": seed,
                "num_threads": num_threads,
            }
            try:
                model = implicit_lib.als.AlternatingLeastSquares(**kwargs)
            except TypeError:
                kwargs.pop("num_threads", None)
                model = implicit_lib.als.AlternatingLeastSquares(**kwargs)
                if hasattr(model, "num_threads"):
                    model.num_threads = num_threads
        elif variant == "MF_IMPLICIT_LMF":
            kwargs = {
                "factors": n_factors,
                "learning_rate": lr_all,
                "regularization": reg_all,
                "iterations": n_epochs,
                "random_state": seed,
                "num_threads": num_threads,
            }
            try:
                model = implicit_lib.lmf.LogisticMatrixFactorization(**kwargs)
            except TypeError:
                kwargs.pop("num_threads", None)
                model = implicit_lib.lmf.LogisticMatrixFactorization(**kwargs)
                if hasattr(model, "num_threads"):
                    model.num_threads = num_threads
        elif variant == "MF_IMPLICIT_BPR":
            kwargs = {
                "factors": n_factors,
                "learning_rate": lr_all,
                "regularization": reg_all,
                "iterations": n_epochs,
                "random_state": seed,
                "num_threads": num_threads,
            }
            try:
                model = implicit_lib.bpr.BayesianPersonalizedRanking(**kwargs)
            except TypeError:
                kwargs.pop("num_threads", None)
                model = implicit_lib.bpr.BayesianPersonalizedRanking(**kwargs)
                if hasattr(model, "num_threads"):
                    model.num_threads = num_threads
        else:
            raise ValueError(f"Unsupported MF variant: {variant}")

        model.fit(user_item, show_progress=False)

        factors = np.asarray(model.item_factors, dtype=np.float32)
        if factors.ndim != 2:
            raise RuntimeError("implicit backend did not return 2D item factors.")
        if factors.shape[0] != user_item.shape[1]:
            full = np.zeros((user_item.shape[1], factors.shape[1]), dtype=np.float32)
            lim = min(user_item.shape[1], factors.shape[0])
            full[:lim] = factors[:lim]
            factors = full
        return factors

    def _train_svd_fallback(self, user_item: sp.csr_matrix) -> np.ndarray:
        """Deterministic fallback when implicit backend is unavailable."""
        n_items = int(user_item.shape[1])
        if n_items == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if user_item.nnz == 0:
            return np.zeros((n_items, 1), dtype=np.float32)

        max_rank = min(user_item.shape[0], user_item.shape[1]) - 1
        n_factors = int(max(1, self.params.get("n_factors", 64)))
        k = min(max_rank, n_factors)
        if k <= 0:
            support = np.asarray(user_item.sum(axis=0)).ravel().astype(np.float32)
            return support.reshape(-1, 1)

        svd = TruncatedSVD(n_components=k, random_state=42)
        item_factors = svd.fit_transform(user_item.T).astype(np.float32)
        if item_factors.ndim == 1:
            item_factors = item_factors.reshape(-1, 1)
        return item_factors

    def fit(self, interactions, **kwargs):
        """
        Fit the MF artifact from exploded interaction logs.

        Accepted kwargs for HRFlow compatibility:
        `targets`, `val_interactions`, `val_targets`, `job_listings`, `all_job_ids`.
        """
        targets = kwargs.get("targets")
        _ = kwargs.get("val_interactions")
        _ = kwargs.get("val_targets")
        _ = kwargs.get("job_listings")
        all_job_ids = kwargs.get("all_job_ids")

        variant = str(self.params.get("variant", "MF_IMPLICIT_BPR")).upper()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported MF variant: {variant}")

        interactions_df = self._normalize_interactions(interactions, "interactions")
        targets_df = self._normalize_targets(targets)

        self._build_job_vocab(interactions_df, targets_df, all_job_ids)
        n_items = len(self.idx_to_job)

        self._item_factors = np.zeros((n_items, 1), dtype=np.float32)
        self._item_factors_unit = self._normalize_rows(self._item_factors)
        self._item_bias = np.zeros((n_items,), dtype=np.float32)
        self._global_bias = 0.0
        self._item_has_signal = np.zeros((n_items,), dtype=bool)

        train_table, n_users = self._build_weighted_interaction_table(
            interactions_df, targets_df
        )
        self._train_item_support = self._build_item_support(train_table, n_items)
        self._build_smoothed_priors(self._train_item_support)

        if n_items == 0 or n_users == 0 or train_table.empty:
            self._backend_name = "empty"
            self.is_fitted = True
            return self

        user_item = self._user_item_matrix(
            train_table, n_users=n_users, n_items=n_items
        )
        if user_item.nnz == 0:
            self._backend_name = "empty"
            self.is_fitted = True
            return self

        try:
            item_factors = self._train_implicit_item_factors(user_item, variant=variant)
            self._backend_name = "implicit"
        except Exception:
            item_factors = self._train_svd_fallback(user_item)
            self._backend_name = "svd_fallback"

        self._item_factors = np.asarray(item_factors, dtype=np.float32)
        self._item_factors_unit = self._normalize_rows(self._item_factors)
        self._item_bias = np.zeros((n_items,), dtype=np.float32)
        self._global_bias = 0.0

        if self._item_factors.shape[0] > 0:
            factor_norms = np.linalg.norm(self._item_factors, axis=1)
            support_ok = self._train_item_support > 0
            self._item_has_signal = (factor_norms > 0) & support_ok
        else:
            self._item_has_signal = np.zeros((n_items,), dtype=bool)

        self.is_fitted = True
        return self

    def _build_query_profile(
        self, hist_items: List[int], hist_actions: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        last_n = int(self.params.get("last_n", 8))
        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        decay = float(self.params.get("query_recency_decay", 0.9))

        items, acts = self._history_slice(hist_items, hist_actions, last_n)
        if not items:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), -1

        length = len(items)
        weight_by_item: Dict[int, float] = defaultdict(float)
        for pos, (item_idx, act) in enumerate(zip(items, acts)):
            w = self._action_weight(act, w_view, w_apply) * (
                decay ** (length - 1 - pos)
            )
            weight_by_item[int(item_idx)] += float(w)

        idxs = np.asarray(sorted(weight_by_item.keys()), dtype=np.int64)
        vals = np.asarray([weight_by_item[idx] for idx in idxs], dtype=np.float32)
        last_item = int(items[-1])
        return idxs, vals, last_item

    def _build_session_vector(
        self, source_factors: np.ndarray, hist_items: List[int], hist_actions: List[str]
    ) -> np.ndarray | None:
        idxs, ws, last_item = self._build_query_profile(hist_items, hist_actions)
        if idxs.size == 0:
            return None

        n_items = int(source_factors.shape[0])
        valid = (idxs >= 0) & (idxs < n_items)
        idxs = idxs[valid]
        ws = ws[valid]
        if idxs.size == 0:
            return None

        vecs = source_factors[idxs]
        if vecs.size == 0:
            return None

        norms = np.linalg.norm(vecs, axis=1)
        keep = norms > 0
        vecs = vecs[keep]
        ws = ws[keep]
        if vecs.shape[0] == 0:
            return None

        mode = str(self.params.get("foldin_mode", "weighted_mean"))
        if mode == "mean":
            user_vec = np.mean(vecs, axis=0)
        else:
            weights = ws.astype(np.float64)
            if float(np.sum(weights)) <= 0:
                weights = np.ones_like(weights, dtype=np.float64)
            user_vec = np.average(vecs, axis=0, weights=weights)

        if mode == "last_item_boost" and 0 <= int(last_item) < n_items:
            boost = float(self.params.get("last_item_boost", 1.2))
            user_vec = user_vec + boost * source_factors[int(last_item)]

        norm = float(np.linalg.norm(user_vec))
        if norm <= 0:
            return None
        return (user_vec / norm).astype(np.float32)

    def _score_candidate_indices(
        self, candidate_indices: np.ndarray, user_vec: np.ndarray
    ) -> np.ndarray:
        if candidate_indices.size == 0:
            return np.array([], dtype=np.float32)

        item_vecs = self._item_factors_unit[candidate_indices]
        scores = np.asarray(item_vecs @ user_vec, dtype=np.float32)

        gamma = float(self.params.get("sim_power_gamma", 1.0))
        if abs(gamma - 1.0) > 1e-9:
            positive = scores > 0
            scores[positive] = np.power(scores[positive], gamma)

        scores = scores + self._item_bias[candidate_indices] + float(self._global_bias)
        return np.asarray(scores, dtype=np.float32)

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """Score all candidate jobs for a single session history."""
        if not self.is_fitted:
            raise RuntimeError("MFModel.predict() called before fit().")

        candidates = [str(job_id) for job_id in candidate_job_ids]
        if not candidates:
            return {}

        hist_df = self._normalize_interactions(session_history, "session_history")
        hist_items: List[int] = []
        hist_actions: List[str] = []
        if not hist_df.empty:
            ordered_hist = hist_df.sort_values("_order", kind="stable")
            for jid, act in zip(
                ordered_hist["job_id"].tolist(), ordered_hist["action"].tolist()
            ):
                idx = self.job_to_idx.get(str(jid), -1)
                if idx < 0:
                    continue
                hist_items.append(int(idx))
                hist_actions.append(str(act))

        user_vec = None
        if hist_items and self._item_factors.shape[0] > 0:
            source_factors = np.asarray(self._item_factors, dtype=np.float32)
            user_vec = self._build_session_vector(
                source_factors, hist_items, hist_actions
            )

        out: Dict[str, float] = {}
        model_score_ids: List[str] = []
        model_score_indices: List[int] = []

        for job_id in candidates:
            idx = self.job_to_idx.get(job_id, -1)
            if idx < 0:
                out[job_id] = float(self.fallback_score)
                continue

            has_signal = bool(
                0 <= idx < len(self._item_has_signal) and self._item_has_signal[idx]
            )
            if user_vec is not None and has_signal:
                model_score_ids.append(job_id)
                model_score_indices.append(int(idx))
            else:
                out[job_id] = float(self._prior_scores.get(job_id, self.fallback_score))

        if model_score_indices and user_vec is not None:
            idx_arr = np.asarray(model_score_indices, dtype=np.int64)
            model_scores = self._score_candidate_indices(idx_arr, user_vec)
            for job_id, score in zip(model_score_ids, model_scores):
                if np.isfinite(score):
                    out[job_id] = float(score)
                else:
                    out[job_id] = float(
                        self._prior_scores.get(job_id, self.fallback_score)
                    )

        return {
            job_id: float(out.get(job_id, self.fallback_score)) for job_id in candidates
        }
