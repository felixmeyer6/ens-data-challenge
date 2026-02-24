"""Content-based lexical+dense ranker."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class CBFModel:
    """Session-conditioned content ranker over job listing text."""

    IS_IMPLEMENTED = True
    REQUIRED_COLUMNS = ("session_id", "job_id", "action")
    SUPPORTED_VARIANTS = {"TFIDF_WORD", "LSA_TFIDF", "LEXICAL_DENSE_BLEND"}
    SUPPORTED_POOLING = {"mean", "recency", "max_sim", "last_item"}
    SUPPORTED_PREPROCESS = {"raw", "normalized", "title_summary"}
    RANDOM_STATE = 42

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "variant": "LEXICAL_DENSE_BLEND",
            "preprocess_mode": "raw",
            "pooling": "recency",
            "w_view": 1.73,
            "w_apply": 2.55,
            "recency_decay": 0.62,
            "blend_alpha_lexical": 0.56,
        }

    @staticmethod
    def suggest_params(trial) -> Dict[str, Any]:
        if trial is None:
            return CBFModel.default_params()
        prefix = "cbf"
        return {
            "variant": "LEXICAL_DENSE_BLEND",
            "preprocess_mode": "raw",
            "pooling": "recency",
            "w_view": trial.suggest_float(f"{prefix}_w_view", 1.6, 1.8),
            "w_apply": trial.suggest_float(f"{prefix}_w_apply", 2.5, 3.0),
            "recency_decay": trial.suggest_float(f"{prefix}_recency_decay", 0.6, 0.65),
            "blend_alpha_lexical": trial.suggest_float(
                f"{prefix}_blend_alpha_lexical", 0.54, 0.58
            ),
        }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = self.default_params()
        if params:
            cfg.update(params)
        self.params = cfg

        self.is_fitted = False
        self.job_to_idx: Dict[str, int] = {}
        self.idx_to_job: List[str] = []

        self._main_sparse: sp.csr_matrix | None = None
        self._main_dense: np.ndarray | None = None
        self._lexical_sparse: sp.csr_matrix | None = None
        self._dense_matrix: np.ndarray | None = None

        self._prior_scores: Dict[str, float] = {}
        self._prior_vector: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._pop_scaled: np.ndarray = np.zeros((0,), dtype=np.float32)
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
        out["action"] = (
            out["action"].map(cls._normalize_action)
            if "action" in out.columns
            else "view"
        )
        out = out.reset_index(drop=True)
        out["_order"] = np.arange(len(out), dtype=np.int64)
        return out[["session_id", "job_id", "action", "_order"]]

    @staticmethod
    def _normalize_text_basic(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.replace("\u200b", " ")
        cleaned = re.sub(r"[_]+x000D[_]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _normalize_text_punctuation(cls, text: Any) -> str:
        cleaned = cls._normalize_text_basic(text).lower()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_title_summary_sections(cls, text: Any) -> str:
        raw = cls._normalize_text_basic(text)
        title_match = re.search(
            r"TITLE\s*(.*?)\s*SUMMARY\s*", raw, flags=re.IGNORECASE | re.DOTALL
        )
        summary_match = re.search(
            r"SUMMARY\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL
        )

        title = title_match.group(1).strip() if title_match else ""
        summary = summary_match.group(1).strip() if summary_match else ""

        if title and summary:
            return f"{title} [SEP] {summary}"
        if summary:
            return summary
        if title:
            return title
        return raw

    @classmethod
    def _preprocess_text(cls, text: Any, mode: str) -> str:
        if mode == "raw":
            return cls._normalize_text_basic(text)
        if mode == "normalized":
            return cls._normalize_text_punctuation(text)
        if mode == "title_summary":
            return cls._normalize_text_punctuation(
                cls._extract_title_summary_sections(text)
            )
        raise ValueError(f"Unknown preprocess_mode: {mode}")

    @staticmethod
    def _safe_minmax(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0:
            return arr.astype(np.float32)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mx - mn < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    @staticmethod
    def _ensure_l2_rows_dense(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32)
        if out.ndim != 2 or out.size == 0:
            return out
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (out / norms).astype(np.float32)

    @staticmethod
    def _action_weight(action: str, w_view: float, w_apply: float) -> float:
        return float(w_apply if action == "apply" else w_view)

    @staticmethod
    def _build_action_recency_weights(
        actions: Sequence[str],
        w_view: float,
        w_apply: float,
        recency_decay: float,
        pooling: str,
    ) -> np.ndarray:
        length = len(actions)
        if length == 0:
            return np.array([], dtype=np.float32)

        act = np.array(
            [w_apply if str(a) == "apply" else w_view for a in actions],
            dtype=np.float64,
        )
        if pooling in {"recency", "max_sim"}:
            decay = np.asarray(
                [float(recency_decay) ** (length - 1 - i) for i in range(length)],
                dtype=np.float64,
            )
        else:
            decay = np.ones(length, dtype=np.float64)

        weights = act * decay
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0:
            weights = np.ones(length, dtype=np.float64) / float(length)
        else:
            weights = weights / weights_sum
        return weights.astype(np.float32)

    @classmethod
    def _score_sparse(
        cls,
        item_matrix: sp.csr_matrix,
        hist_idxs: List[int],
        hist_actions: List[str],
        w_view: float,
        w_apply: float,
        recency_decay: float,
        pooling: str,
    ) -> np.ndarray:
        n_items_local = int(item_matrix.shape[0])
        if len(hist_idxs) == 0 or n_items_local == 0 or int(item_matrix.shape[1]) == 0:
            return np.zeros(n_items_local, dtype=np.float32)

        weights = cls._build_action_recency_weights(
            hist_actions, w_view, w_apply, recency_decay, pooling
        )

        if pooling in {"mean", "recency"}:
            profile = (
                sp.csr_matrix(weights.reshape(1, -1), dtype=np.float32)
                @ item_matrix[hist_idxs]
            )
            scores = (profile @ item_matrix.T).toarray().ravel()
            return np.asarray(scores, dtype=np.float32)

        if pooling == "last_item":
            last = int(hist_idxs[-1])
            scores = (item_matrix[last] @ item_matrix.T).toarray().ravel()
            return np.asarray(scores, dtype=np.float32)

        if pooling == "max_sim":
            acc = np.full(n_items_local, -np.inf, dtype=np.float64)
            for h_idx, w in zip(hist_idxs, weights):
                vec = (item_matrix[int(h_idx)] @ item_matrix.T).toarray().ravel()
                acc = np.maximum(acc, float(w) * vec)
            return np.asarray(acc, dtype=np.float32)

        raise ValueError(f"Unknown pooling mode: {pooling}")

    @classmethod
    def _score_dense(
        cls,
        item_matrix: np.ndarray,
        hist_idxs: List[int],
        hist_actions: List[str],
        w_view: float,
        w_apply: float,
        recency_decay: float,
        pooling: str,
    ) -> np.ndarray:
        n_items_local = int(item_matrix.shape[0])
        if len(hist_idxs) == 0 or n_items_local == 0:
            return np.zeros(n_items_local, dtype=np.float32)

        hist = item_matrix[np.asarray(hist_idxs, dtype=np.int64)]
        weights = cls._build_action_recency_weights(
            hist_actions, w_view, w_apply, recency_decay, pooling
        )

        if pooling in {"mean", "recency"}:
            profile = np.sum(hist * weights[:, None], axis=0)
            norm = float(np.linalg.norm(profile))
            if norm > 1e-12:
                profile = profile / norm
            return np.asarray(item_matrix @ profile, dtype=np.float32)

        if pooling == "last_item":
            return np.asarray(item_matrix @ hist[-1], dtype=np.float32)

        if pooling == "max_sim":
            sims = item_matrix @ hist.T  # [n_items, L]
            sims = sims * weights[None, :]
            return np.asarray(np.max(sims, axis=1), dtype=np.float32)

        raise ValueError(f"Unknown pooling mode: {pooling}")

    @staticmethod
    def _extract_listing_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in (
                "text",
                "description",
                "summary",
                "content",
                "job_description",
                "title",
            ):
                if key in value and value[key] is not None:
                    return str(value[key])
            parts = [str(v) for v in value.values() if isinstance(v, (str, int, float))]
            return " ".join(parts).strip()
        if isinstance(value, (list, tuple)):
            parts = [str(v) for v in value if isinstance(v, (str, int, float))]
            return " ".join(parts).strip()
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _coerce_job_listings(cls, job_listings: Any) -> Dict[str, str]:
        """Normalize job listings into a map: job_id(str) -> text."""
        if job_listings is None:
            return {}

        if isinstance(job_listings, pd.DataFrame):
            if len(job_listings) == 0:
                return {}
            id_col = None
            for candidate in ("job_id", "id", "listing_id", "key"):
                if candidate in job_listings.columns:
                    id_col = candidate
                    break
            if id_col is None:
                id_series = job_listings.index.to_series()
            else:
                id_series = job_listings[id_col]

            text_col = None
            for candidate in (
                "text",
                "description",
                "summary",
                "content",
                "job_description",
                "title",
            ):
                if candidate in job_listings.columns:
                    text_col = candidate
                    break

            out: Dict[str, str] = {}
            for idx, jid in id_series.items():
                if pd.isna(jid):
                    continue
                if text_col is not None:
                    value = job_listings.at[idx, text_col]
                else:
                    value = job_listings.loc[idx].to_dict()
                out[str(jid)] = cls._extract_listing_text(value)
            return out

        if isinstance(job_listings, Mapping):
            out: Dict[str, str] = {}
            for jid, value in job_listings.items():
                out[str(jid)] = cls._extract_listing_text(value)
            return out

        if isinstance(job_listings, (list, tuple)):
            out = {}
            for row in job_listings:
                if not isinstance(row, Mapping):
                    continue
                jid = None
                for key in ("job_id", "id", "listing_id", "key"):
                    if key in row and row[key] is not None:
                        jid = str(row[key])
                        break
                if jid is None:
                    continue
                out[jid] = cls._extract_listing_text(row)
            return out

        return {}

    @staticmethod
    def _build_tfidf_word(
        texts: List[str],
        params: Dict[str, Any],
    ) -> sp.csr_matrix:
        n_items = len(texts)
        if n_items == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        try:
            vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, int(params.get("word_ngram_max", 2))),
                min_df=int(max(1, params.get("min_df", 2))),
                max_df=float(params.get("max_df", 0.95)),
                max_features=int(params.get("max_features", 30000)),
                sublinear_tf=True,
                lowercase=bool(params.get("lowercase", True)),
                strip_accents=params.get("strip_accents", "unicode"),
                token_pattern=params.get("token_pattern", r"(?u)\b\w\w+\b"),
            )
            x = vec.fit_transform(texts)
            if x.shape[1] == 0:
                return sp.csr_matrix((n_items, 0), dtype=np.float32)
            x = normalize(x, norm="l2")
            return x.tocsr().astype(np.float32)
        except ValueError:
            # Handles edge cases such as empty vocabulary.
            return sp.csr_matrix((n_items, 0), dtype=np.float32)

    @classmethod
    def _build_lsa_tfidf_dense(
        cls,
        texts: List[str],
        params: Dict[str, Any],
    ) -> np.ndarray:
        n_items = len(texts)
        if n_items == 0:
            return np.zeros((0, 1), dtype=np.float32)

        sparse_x = cls._build_tfidf_word(texts, params)
        if sparse_x.shape[1] <= 1:
            return np.zeros((n_items, 1), dtype=np.float32)

        wanted = int(params.get("lsa_components", 256))
        n_components = max(1, min(wanted, int(sparse_x.shape[1]) - 1))

        try:
            svd = TruncatedSVD(n_components=n_components, random_state=cls.RANDOM_STATE)
            dense = svd.fit_transform(sparse_x).astype(np.float32)
            return cls._ensure_l2_rows_dense(dense)
        except ValueError:
            return np.zeros((n_items, 1), dtype=np.float32)

    def _build_prior_scores(
        self,
        interactions_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> None:
        n_items = len(self.idx_to_job)
        if n_items == 0:
            self._prior_scores = {}
            self._prior_vector = np.zeros((0,), dtype=np.float32)
            self._pop_scaled = np.zeros((0,), dtype=np.float32)
            self.fallback_score = 0.0
            return

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        alpha = 1.0

        raw = np.zeros((n_items,), dtype=np.float64)
        for df in (interactions_df, targets_df):
            if df.empty:
                continue
            for jid, action in zip(df["job_id"].tolist(), df["action"].tolist()):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                raw[idx] += self._action_weight(str(action), w_view, w_apply)

        total = float(np.sum(raw))
        denom = total + alpha * float(n_items)
        if denom <= 0:
            self._prior_scores = {jid: 1.0 for jid in self.idx_to_job}
            self._prior_vector = np.ones((n_items,), dtype=np.float32)
            self._pop_scaled = np.ones((n_items,), dtype=np.float32)
            self.fallback_score = 1.0
            return

        smoothed = (raw + alpha) / denom
        self._prior_vector = smoothed.astype(np.float32)
        self._pop_scaled = self._safe_minmax(self._prior_vector)
        self._prior_scores = {
            jid: float(self._prior_vector[idx])
            for idx, jid in enumerate(self.idx_to_job)
        }
        self.fallback_score = float(alpha / denom)

    def fit(self, interactions, **kwargs):
        """
        Fit CBF artifacts from exploded interaction logs.

        Accepted kwargs for HRFlow compatibility:
        `targets`, `val_interactions`, `val_targets`, `job_listings`, `all_job_ids`.
        """
        targets = kwargs.get("targets")
        _ = kwargs.get("val_interactions")
        _ = kwargs.get("val_targets")
        job_listings = kwargs.get("job_listings")
        all_job_ids: Sequence[Any] | None = kwargs.get("all_job_ids")

        variant = str(self.params.get("variant", "LEXICAL_DENSE_BLEND")).upper()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported CBF variant: {variant}")

        preprocess_mode = str(self.params.get("preprocess_mode", "raw")).lower()
        if preprocess_mode not in self.SUPPORTED_PREPROCESS:
            raise ValueError(
                f"Unsupported preprocess_mode: {preprocess_mode}. "
                f"Supported modes: {sorted(self.SUPPORTED_PREPROCESS)}"
            )

        pooling = str(self.params.get("pooling", "recency")).lower()
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(
                f"Unsupported pooling mode: {pooling}. "
                f"Supported modes: {sorted(self.SUPPORTED_POOLING)}"
            )

        interactions_df = self._normalize_interactions(interactions, "interactions")
        targets_df = self._normalize_targets(targets)
        text_by_job = self._coerce_job_listings(job_listings)

        extra_ids = (
            [str(x) for x in all_job_ids if pd.notna(x)]
            if all_job_ids is not None
            else []
        )
        known_ids = self._ordered_unique(
            extra_ids
            + interactions_df["job_id"].drop_duplicates().tolist()
            + targets_df["job_id"].drop_duplicates().tolist()
            + list(text_by_job.keys())
        )

        self.idx_to_job = list(known_ids)
        self.job_to_idx = {jid: idx for idx, jid in enumerate(self.idx_to_job)}

        item_texts = [
            self._preprocess_text(text_by_job.get(jid, ""), preprocess_mode)
            for jid in self.idx_to_job
        ]

        self._main_sparse = None
        self._main_dense = None
        self._lexical_sparse = None
        self._dense_matrix = None

        if variant == "TFIDF_WORD":
            main_cfg = {
                "max_features": self.params.get("max_features", 30000),
                "min_df": self.params.get("min_df", 2),
                "max_df": self.params.get("max_df", 0.95),
                "word_ngram_max": self.params.get("word_ngram_max", 2),
                "token_pattern": self.params.get("tokenizer", {}).get(
                    "token_pattern", r"(?u)\b\w\w+\b"
                ),
                "strip_accents": self.params.get("tokenizer", {}).get(
                    "strip_accents", "unicode"
                ),
                "lowercase": self.params.get("tokenizer", {}).get("lowercase", True),
            }
            self._main_sparse = self._build_tfidf_word(item_texts, main_cfg)

        elif variant == "LSA_TFIDF":
            main_cfg = {
                "max_features": self.params.get("max_features", 30000),
                "min_df": self.params.get("min_df", 2),
                "max_df": self.params.get("max_df", 0.95),
                "word_ngram_max": self.params.get("word_ngram_max", 2),
                "lsa_components": self.params.get("lsa_components", 256),
                "token_pattern": self.params.get("tokenizer", {}).get(
                    "token_pattern", r"(?u)\b\w\w+\b"
                ),
                "strip_accents": self.params.get("tokenizer", {}).get(
                    "strip_accents", "unicode"
                ),
                "lowercase": self.params.get("tokenizer", {}).get("lowercase", True),
            }
            self._main_dense = self._build_lsa_tfidf_dense(item_texts, main_cfg)

        else:  # LEXICAL_DENSE_BLEND
            lex_cfg = {
                "max_features": self.params.get("max_features", 30000),
                "min_df": self.params.get("min_df", 2),
                "max_df": self.params.get("max_df", 0.95),
                "word_ngram_max": self.params.get("lex_word_ngram_max", 2),
                "token_pattern": self.params.get("tokenizer", {}).get(
                    "token_pattern", r"(?u)\b\w\w+\b"
                ),
                "strip_accents": self.params.get("tokenizer", {}).get(
                    "strip_accents", "unicode"
                ),
                "lowercase": self.params.get("tokenizer", {}).get("lowercase", True),
            }
            dense_cfg = {
                "max_features": self.params.get("max_features", 30000),
                "min_df": self.params.get("min_df", 2),
                "max_df": self.params.get("max_df", 0.95),
                "word_ngram_max": self.params.get("dense_word_ngram_max", 2),
                "lsa_components": self.params.get("dense_lsa_components", 256),
                "token_pattern": self.params.get("tokenizer", {}).get(
                    "token_pattern", r"(?u)\b\w\w+\b"
                ),
                "strip_accents": self.params.get("tokenizer", {}).get(
                    "strip_accents", "unicode"
                ),
                "lowercase": self.params.get("tokenizer", {}).get("lowercase", True),
            }
            self._lexical_sparse = self._build_tfidf_word(item_texts, lex_cfg)
            self._dense_matrix = self._build_lsa_tfidf_dense(item_texts, dense_cfg)

        self._build_prior_scores(interactions_df, targets_df)
        self.is_fitted = True
        return self

    def _score_known_items(
        self, hist_idxs: List[int], hist_actions: List[str]
    ) -> np.ndarray:
        n_items = len(self.idx_to_job)
        if n_items == 0 or len(hist_idxs) == 0:
            return np.zeros((n_items,), dtype=np.float32)

        w_view = float(self.params.get("w_view", 1.0))
        w_apply = float(self.params.get("w_apply", 2.0))
        recency_decay = float(self.params.get("recency_decay", 1.0))
        pooling = str(self.params.get("pooling", "recency")).lower()
        variant = str(self.params.get("variant", "LEXICAL_DENSE_BLEND")).upper()

        if variant == "LEXICAL_DENSE_BLEND":
            lex_scores = (
                self._score_sparse(
                    self._lexical_sparse,
                    hist_idxs,
                    hist_actions,
                    w_view=w_view,
                    w_apply=w_apply,
                    recency_decay=recency_decay,
                    pooling=pooling,
                )
                if self._lexical_sparse is not None
                else np.zeros((n_items,), dtype=np.float32)
            )
            dense_scores = (
                self._score_dense(
                    self._dense_matrix,
                    hist_idxs,
                    hist_actions,
                    w_view=w_view,
                    w_apply=w_apply,
                    recency_decay=recency_decay,
                    pooling=pooling,
                )
                if self._dense_matrix is not None
                else np.zeros((n_items,), dtype=np.float32)
            )

            if bool(self.params.get("normalize_component_scores", True)):
                lex_scores = self._safe_minmax(lex_scores)
                dense_scores = self._safe_minmax(dense_scores)

            alpha_lex = float(self.params.get("blend_alpha_lexical", 0.5))
            scores = alpha_lex * lex_scores + (1.0 - alpha_lex) * dense_scores
        elif variant == "TFIDF_WORD":
            scores = self._score_sparse(
                self._main_sparse,
                hist_idxs,
                hist_actions,
                w_view=w_view,
                w_apply=w_apply,
                recency_decay=recency_decay,
                pooling=pooling,
            )
        else:
            scores = self._score_dense(
                self._main_dense,
                hist_idxs,
                hist_actions,
                w_view=w_view,
                w_apply=w_apply,
                recency_decay=recency_decay,
                pooling=pooling,
            )

        pop_w = float(self.params.get("popularity_weight", 0.0))
        if pop_w > 0 and self._pop_scaled.size == scores.size:
            scores = (1.0 - pop_w) * np.asarray(scores) + pop_w * self._pop_scaled
        return np.asarray(scores, dtype=np.float32)

    def predict(self, session_history, candidate_job_ids: Iterable) -> Dict[str, float]:
        """
        Score candidates for one session.

        Returns a score for every candidate id passed in. Seen-item filtering is
        intentionally not applied here (hybrid layer handles filtering/ranking).
        """
        if not self.is_fitted:
            raise RuntimeError("CBFModel must be fitted before calling predict().")
        if candidate_job_ids is None:
            return {}

        hist_df = self._normalize_interactions(session_history, "session_history")
        hist_df = hist_df.sort_values("_order", kind="stable")

        hist_idxs: List[int] = []
        hist_actions: List[str] = []
        if not hist_df.empty:
            for jid, action in zip(
                hist_df["job_id"].tolist(), hist_df["action"].tolist()
            ):
                idx = self.job_to_idx.get(str(jid))
                if idx is None:
                    continue
                hist_idxs.append(int(idx))
                hist_actions.append(str(action))

        signal_scores = self._score_known_items(hist_idxs, hist_actions)

        out: Dict[str, float] = {}
        for candidate in candidate_job_ids:
            jid = str(candidate)
            idx = self.job_to_idx.get(jid)

            if idx is not None and idx < signal_scores.size:
                signal = float(signal_scores[idx])
                if np.isfinite(signal) and signal > 0:
                    out[jid] = signal
                    continue
                out[jid] = float(self._prior_scores.get(jid, self.fallback_score))
                continue

            out[jid] = float(self._prior_scores.get(jid, self.fallback_score))

        return out
