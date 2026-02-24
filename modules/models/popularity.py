"""Popularity ranker."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


class PopularityModel:
    """Popularity ranker ported from POC #0 (session-weighted default variant)."""

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    SUPPORTED_VARIANTS = {
        "POP_GLOBAL_COUNTS",
        "POP_SESSION_WEIGHTED",
        "POP_TRENDING",
        "POP_HYBRID",
    }

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "variant": "POP_SESSION_WEIGHTED",
            "include_target": False,
            "normalize_scores": True,
            "recency_decay": 0.5825575752466695,
            "smooth_alpha": 1.6483148299436956,
            "w_apply": 6.335092003805592,
            "w_view": 0.9299464263001853,
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return PopularityModel.default_params()
        return {
            "variant": "POP_SESSION_WEIGHTED",
            "include_target": False,
            "normalize_scores": True,
            "recency_decay": trial.suggest_float("pop_recency_decay", 0.51, 0.59),
            "smooth_alpha": trial.suggest_float("pop_smooth_alpha", 0.60, 3.0),
            "w_apply": trial.suggest_float("pop_w_apply", 5.0, 7.0),
            "w_view": trial.suggest_float("pop_w_view", 0.9, 1.5),
        }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = cfg
        self.is_fitted = False
        self.item_scores: Dict[str, float] = {}
        self.fallback_score: float = 0.0

    @staticmethod
    def _normalize_action(value: Any) -> str:
        return str(value).strip().lower()

    @classmethod
    def _normalize_interactions(cls, interactions: pd.DataFrame | None) -> pd.DataFrame:
        """Validate and normalize interactions to canonical string columns."""
        if interactions is None or len(interactions) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in interactions.columns]
        if missing:
            raise ValueError(
                f"PopularityModel.fit() missing required interaction columns: {missing}"
            )

        out = interactions.loc[:, list(cls.REQUIRED_COLUMNS)].copy()
        out = out.dropna(subset=list(cls.REQUIRED_COLUMNS))
        out["session_id"] = out["session_id"].map(str)
        out["job_id"] = out["job_id"].map(str)
        out["action"] = out["action"].map(cls._normalize_action)
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out

    @staticmethod
    def _normalize_targets(targets: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize target rows (used when include_target=True)."""
        if targets is None or len(targets) == 0:
            return pd.DataFrame(columns=["session_id", "job_id", "action", "_order"])

        for col in ("session_id", "job_id"):
            if col not in targets.columns:
                raise ValueError(
                    f"PopularityModel.fit() missing required target column: '{col}'"
                )

        out = targets.copy()
        out = out.dropna(subset=["session_id", "job_id"])
        out["session_id"] = out["session_id"].map(str)
        out["job_id"] = out["job_id"].map(str)
        if "action" in out.columns:
            out["action"] = out["action"].map(PopularityModel._normalize_action)
        else:
            out["action"] = "view"
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out[["session_id", "job_id", "action", "_order"]]

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> List[str]:
        """Preserve first-seen order while removing duplicates."""
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

    def _compute_global_counts(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
    ) -> pd.Series:
        scores = (
            interactions.groupby("job_id", sort=False).size().astype(np.float64)
            if not interactions.empty
            else pd.Series(dtype=np.float64)
        )

        if include_target and not targets.empty:
            target_counts = (
                targets.groupby("job_id", sort=False).size().astype(np.float64)
            )
            scores = scores.add(target_counts, fill_value=0.0)
        return scores

    def _compute_session_weighted(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
        w_view: float,
        w_apply: float,
        recency_decay: float,
    ) -> pd.Series:
        if interactions.empty:
            scores = pd.Series(dtype=np.float64)
        else:
            work = interactions.sort_values("_order", kind="stable").copy()
            group = work.groupby("session_id", sort=False)
            work["_pos"] = group.cumcount().astype(np.int64)
            work["_len"] = group["job_id"].transform("size").astype(np.int64)

            action_weights = np.where(
                work["action"].to_numpy() == "apply", float(w_apply), float(w_view)
            )
            recency_steps = (work["_len"] - work["_pos"] - 1).to_numpy(dtype=np.int64)
            recency_weights = np.power(float(recency_decay), recency_steps)
            work["_event_w"] = action_weights * recency_weights

            scores = (
                work.groupby("job_id", sort=False)["_event_w"].sum().astype(np.float64)
            )

        if include_target and not targets.empty:
            target_boost = max(float(w_apply), 1.0)
            target_counts = (
                targets.groupby("job_id", sort=False).size().astype(np.float64)
            )
            scores = scores.add(target_counts * target_boost, fill_value=0.0)
        return scores

    def _compute_trending(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
        halflife: float,
    ) -> pd.Series:
        if interactions.empty and targets.empty:
            return pd.Series(dtype=np.float64)

        ordered_sessions: List[str] = []
        if not interactions.empty:
            ordered_sessions.extend(
                interactions.sort_values("_order", kind="stable")
                .drop_duplicates(subset=["session_id"], keep="first")["session_id"]
                .tolist()
            )
        if not targets.empty:
            target_sessions = (
                targets.sort_values("_order", kind="stable")
                .drop_duplicates(subset=["session_id"], keep="first")["session_id"]
                .tolist()
            )
            seen = set(ordered_sessions)
            ordered_sessions.extend([sid for sid in target_sessions if sid not in seen])

        if not ordered_sessions:
            return pd.Series(dtype=np.float64)

        rank_map = {sid: idx for idx, sid in enumerate(ordered_sessions)}
        n_sessions = float(len(ordered_sessions))
        safe_halflife = max(float(halflife), 1e-12)

        history_scores = pd.Series(dtype=np.float64)
        if not interactions.empty:
            dedup = interactions.sort_values("_order", kind="stable").drop_duplicates(
                subset=["session_id", "job_id"], keep="first"
            )
            rank = dedup["session_id"].map(rank_map).to_numpy(dtype=np.float64)
            time_w = np.exp((-np.log(2.0) * (n_sessions - rank)) / safe_halflife)
            dedup = dedup.assign(_time_w=time_w)
            history_scores = (
                dedup.groupby("job_id", sort=False)["_time_w"].sum().astype(np.float64)
            )

        if include_target and not targets.empty:
            target_rank = targets["session_id"].map(rank_map).to_numpy(dtype=np.float64)
            target_w = np.exp(
                (-np.log(2.0) * (n_sessions - target_rank)) / safe_halflife
            )
            target_df = targets.assign(_time_w=target_w * 1.5)
            target_scores = (
                target_df.groupby("job_id", sort=False)["_time_w"]
                .sum()
                .astype(np.float64)
            )
            history_scores = history_scores.add(target_scores, fill_value=0.0)

        return history_scores

    @staticmethod
    def _max_normalize(scores: pd.Series) -> pd.Series:
        if scores.empty:
            return scores
        max_score = float(scores.max())
        if max_score > 0:
            return scores / max_score
        return scores

    def _compute_variant_scores(
        self,
        interactions: pd.DataFrame,
        targets: pd.DataFrame,
        include_target: bool,
    ) -> pd.Series:
        variant = str(self.params.get("variant", "POP_SESSION_WEIGHTED")).upper()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unknown popularity variant: {variant}")

        if variant == "POP_GLOBAL_COUNTS":
            return self._compute_global_counts(interactions, targets, include_target)

        if variant == "POP_SESSION_WEIGHTED":
            return self._compute_session_weighted(
                interactions=interactions,
                targets=targets,
                include_target=include_target,
                w_view=float(self.params.get("w_view", 1.0)),
                w_apply=float(self.params.get("w_apply", 2.0)),
                recency_decay=float(self.params.get("recency_decay", 0.85)),
            )

        if variant == "POP_TRENDING":
            return self._compute_trending(
                interactions=interactions,
                targets=targets,
                include_target=include_target,
                halflife=float(self.params.get("trending_halflife", 100.0)),
            )

        alpha = float(self.params.get("hybrid_alpha", 0.7))
        global_scores = self._max_normalize(
            self._compute_global_counts(interactions, targets, include_target)
        )
        trending_scores = self._max_normalize(
            self._compute_trending(
                interactions=interactions,
                targets=targets,
                include_target=include_target,
                halflife=float(self.params.get("trending_halflife", 100.0)),
            )
        )
        return global_scores.mul(alpha).add(
            trending_scores.mul(1.0 - alpha), fill_value=0.0
        )

    def fit(self, interactions, **kwargs):
        """Fit job popularity scores from exploded interaction logs."""
        targets = kwargs.get("targets")
        all_job_ids: Sequence[Any] | None = kwargs.get("all_job_ids")
        include_target = bool(self.params.get("include_target", False))

        interactions_df = self._normalize_interactions(interactions)
        targets_df = (
            self._normalize_targets(targets) if include_target else pd.DataFrame()
        )

        extra_ids = (
            [str(x) for x in all_job_ids if pd.notna(x)]
            if all_job_ids is not None
            else []
        )
        known_ids = self._ordered_unique(
            extra_ids
            + interactions_df["job_id"].drop_duplicates().tolist()
            + (
                targets_df["job_id"].drop_duplicates().tolist()
                if not targets_df.empty
                else []
            )
        )

        raw_scores = self._compute_variant_scores(
            interactions=interactions_df,
            targets=targets_df,
            include_target=include_target,
        )

        scores = pd.Series(0.0, index=known_ids, dtype=np.float64)
        scores = scores.add(raw_scores, fill_value=0.0)

        smooth_alpha = float(self.params.get("smooth_alpha", 0.0))
        scores = scores + smooth_alpha

        normalize = bool(self.params.get("normalize_scores", True))
        max_score = float(scores.max()) if not scores.empty else smooth_alpha
        fallback = float(smooth_alpha)

        if normalize and max_score > 0:
            scores = scores / max_score
            fallback = fallback / max_score

        self.item_scores = {
            str(job_id): float(score) for job_id, score in scores.items()
        }
        self.fallback_score = float(fallback)
        self.is_fitted = True
        return self

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """
        Score all candidate jobs with learned popularity (or smoothing fallback).

        The model is global and does not remove seen items; filtering is delegated
        to the hybrid/ranking layer.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "PopularityModel must be fitted before calling predict()."
            )

        if candidate_job_ids is None:
            return {}

        scores: Dict[str, float] = {}
        for job_id in candidate_job_ids:
            key = str(job_id)
            scores[key] = float(self.item_scores.get(key, self.fallback_score))
        return scores
