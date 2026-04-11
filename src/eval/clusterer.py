from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

from metrics.mojo import MoJoCalculator
from metrics.a2a import A2ACalculator
from metrics.c2c import C2CCoverage
from metrics.turbomq import TurboMQ


class ClusterAndEval:
    def __init__(
        self,
        k_range: Iterable[int] = range(10, 31),
        sample_size: int = 2000,
        ahc_linkage: str = "ward",
    ):
        self.k_range = list(k_range)
        self.sample_size = int(sample_size)
        self.ahc_linkage = str(ahc_linkage)

    def _get_nodes(self, data, n: int) -> List[str]:
        if hasattr(data, "nodes") and data.nodes is not None and len(data.nodes) == n:
            return [str(x) for x in list(data.nodes)]

        if hasattr(data, "node_list") and data.node_list is not None and len(data.node_list) == n:
            return [str(x) for x in list(data.node_list)]

        if (
            hasattr(data, "df")
            and isinstance(data.df, pd.DataFrame)
            and "File" in data.df.columns
            and len(data.df) == n
        ):
            return data.df["File"].astype(str).tolist()

        return [str(i) for i in range(n)]

    def _get_y_true(self, data, n: int) -> Optional[np.ndarray]:
        if hasattr(data, "y_true") and data.y_true is not None and len(data.y_true) == n:
            return np.asarray(data.y_true, dtype=int)

        if (
            hasattr(data, "df")
            and isinstance(data.df, pd.DataFrame)
            and "Label" in data.df.columns
            and len(data.df) == n
        ):
            return data.df["Label"].to_numpy(dtype=int)

        return None

    def _get_deps(self, data) -> Optional[pd.DataFrame]:
        if hasattr(data, "df_dep") and isinstance(data.df_dep, pd.DataFrame):
            return data.df_dep
        return None

    def _get_eval_frame(self, data, nodes: List[str]) -> Optional[pd.DataFrame]:
        """
        Align data.df to the exact node order used by preds / y_true.
        This is crucial for NEGARData, where df row order may differ from node_list order.
        """
        if not hasattr(data, "df") or not isinstance(data.df, pd.DataFrame):
            return None

        needed = {"File", "Primary_Module", "Module_List", "Duplicated"}
        if not needed.issubset(set(data.df.columns)):
            return None

        df = data.df.copy()
        df["File"] = df["File"].astype(str)

        aligned = (
            df.set_index("File")
            .reindex(nodes)
            .reset_index()
            .rename(columns={"index": "File"})
        )

        return aligned

    def _valid_ks(self, n: int) -> List[int]:
        """
        Keep only valid k values for the current dataset size.
        Example: if n=20, then k=30 is impossible and should be removed.
        """
        ks = sorted({int(k) for k in self.k_range if 1 < int(k) < n})
        if not ks:
            raise ValueError(f"No valid k in k_range for n={n}")
        return ks

    def _fit_labels(self, X: np.ndarray, k: int, clustering: str) -> np.ndarray:
        k = int(k)
        c = str(clustering).lower()

        if c == "kmeans":
            return KMeans(n_clusters=k).fit_predict(X)

        if c in {"ahc", "agg", "agglomerative"}:
            return AgglomerativeClustering(
                n_clusters=k,
                linkage=self.ahc_linkage,
            ).fit_predict(X)

        raise ValueError(f"Unknown clustering='{clustering}'. Use 'kmeans' or 'ahc'.")

    def _best_k_elbow_kmeans(self, X: np.ndarray) -> Tuple[int, pd.DataFrame]:
        n = X.shape[0]
        ks = self._valid_ks(n)

        m = min(self.sample_size, n)
        idx = np.random.choice(n, size=m, replace=False)
        Xs = X[idx]

        inertias: List[float] = []
        for k in ks:
            km = KMeans(n_clusters=int(k)).fit(Xs)
            inertias.append(float(km.inertia_))

        if len(ks) <= 2:
            return int(ks[0]), pd.DataFrame({"k": ks, "inertia": inertias})

        x = np.array(ks, dtype=float)
        y = np.array(inertias, dtype=float)

        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)

        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        v = p2 - p1
        v = v / (np.linalg.norm(v) + 1e-12)

        pts = np.stack([x, y], axis=1)
        proj = p1 + ((pts - p1) @ v)[:, None] * v[None, :]
        d = np.linalg.norm(pts - proj, axis=1)

        best_k = int(ks[int(np.argmax(d))])
        diag = pd.DataFrame({"k": ks, "inertia": inertias, "elbow_dist": d.tolist()})
        return best_k, diag

    def _best_k_elbow_ahc(self, X: np.ndarray) -> Tuple[int, pd.DataFrame]:
        n = X.shape[0]
        ks = self._valid_ks(n)

        m = min(self.sample_size, n)
        idx = np.random.choice(n, size=m, replace=False)
        Xs = X[idx]

        wcss_vals: List[float] = []
        for k in ks:
            labels = AgglomerativeClustering(
                n_clusters=int(k),
                linkage=self.ahc_linkage,
            ).fit_predict(Xs)

            w = 0.0
            for c in np.unique(labels):
                Xc = Xs[labels == c]
                if Xc.shape[0] <= 1:
                    continue
                mu = Xc.mean(axis=0, keepdims=True)
                diff = Xc - mu
                w += float((diff * diff).sum())

            wcss_vals.append(w)

        if len(ks) <= 2:
            return int(ks[0]), pd.DataFrame({"k": ks, "wcss": wcss_vals})

        x = np.array(ks, dtype=float)
        y = np.array(wcss_vals, dtype=float)

        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)

        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        v = p2 - p1
        v = v / (np.linalg.norm(v) + 1e-12)

        pts = np.stack([x, y], axis=1)
        proj = p1 + ((pts - p1) @ v)[:, None] * v[None, :]
        d = np.linalg.norm(pts - proj, axis=1)

        best_k = int(ks[int(np.argmax(d))])
        diag = pd.DataFrame({"k": ks, "wcss": wcss_vals, "elbow_dist": d.tolist()})
        return best_k, diag

    def _choose_k(self, X: np.ndarray, clustering: str) -> Tuple[int, pd.DataFrame]:
        c = str(clustering).lower()

        if c in {"ahc", "agg", "agglomerative"}:
            return self._best_k_elbow_ahc(X)
        if c == "kmeans":
            return self._best_k_elbow_kmeans(X)
        raise ValueError(f"Unknown clustering='{clustering}'. Use 'kmeans' or 'ahc'.")

    def _cluster_to_module_map(self, preds: np.ndarray, eval_df: pd.DataFrame, data) -> Dict[int, str]:
        """
        Assign one predicted module label to each predicted cluster.
        We use majority vote over Primary_Module, preferring non-duplicated files.
        """
        duplicated = eval_df["Duplicated"].fillna(False).astype(bool).to_numpy()
        primary_modules = (
            eval_df["Primary_Module"]
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        cluster_to_module: Dict[int, str] = {}

        for c in np.unique(preds):
            idx = np.where(preds == c)[0]

            vote_pool = [primary_modules[i] for i in idx if not duplicated[i]]
            if not vote_pool:
                vote_pool = [primary_modules[i] for i in idx]

            if not vote_pool:
                continue

            counts = pd.Series(vote_pool).value_counts()
            best_count = counts.max()
            best_labels = sorted([lbl for lbl, cnt in counts.items() if cnt == best_count])
            cluster_to_module[int(c)] = best_labels[0]

        return cluster_to_module

    def _apply_multilabel_relaxation(
        self,
        preds: np.ndarray,
        y_true: np.ndarray,
        nodes: List[str],
        data,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Exact requested behavior:

        For duplicated files, if the predicted cluster label (mapped to a module label)
        is in Module_List, then count that file as correct by replacing its GT label with
        that predicted module label.

        Non-duplicated files remain unchanged.
        """
        y = np.asarray(y_true, dtype=int).copy()
        out: Dict[str, float] = {}

        if not hasattr(data, "label_encoder") or data.label_encoder is None:
            return y, out

        eval_df = self._get_eval_frame(data, nodes)
        if eval_df is None:
            return y, out

        enc_map = {str(cls).strip().lower(): i for i, cls in enumerate(data.label_encoder.classes_)}

        preds = np.asarray(preds, dtype=int)
        duplicated = eval_df["Duplicated"].fillna(False).astype(bool).to_numpy()

        module_lists: List[List[str]] = []
        for val in eval_df["Module_List"].tolist():
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                module_lists.append(
                    [str(x).strip().lower() for x in val if pd.notna(x) and str(x).strip()]
                )
            elif pd.isna(val):
                module_lists.append([])
            else:
                module_lists.append([str(val).strip().lower()])

        cluster_to_module = self._cluster_to_module_map(preds, eval_df, data)

        if duplicated.any():
            dup_idx = np.where(duplicated)[0]
            hits: List[bool] = []
            adjusted = 0

            for i in dup_idx:
                pred_module = cluster_to_module.get(int(preds[i]))
                allowed_modules = module_lists[i]

                if pred_module is not None and pred_module in allowed_modules and pred_module in enc_map:
                    y[i] = enc_map[pred_module]
                    hits.append(True)
                    adjusted += 1
                else:
                    hits.append(False)

            if hits:
                out["Duplicated_HitAny"] = 100.0 * float(np.mean(hits))
                out["Duplicated_Adjusted"] = float(adjusted)

        return y, out

    def _eval(
        self,
        labels: np.ndarray,
        y_true: np.ndarray,
        nodes: List[str],
        deps: Optional[pd.DataFrame],
        data,
    ) -> Dict[str, float]:
        preds, _ = pd.factorize(labels)

        y_adj, extra = self._apply_multilabel_relaxation(
            preds=preds,
            y_true=y_true,
            nodes=nodes,
            data=data,
        )
        out: Dict[str, float] = {}
        out.update(
            {
                "MoJoFM": MoJoCalculator(preds, y_adj, mode="array").mojofm(),
                "A2A": A2ACalculator(preds, y_adj, mode="array").a2a(),
                "C2CCvg_10": C2CCoverage((nodes, preds), (nodes, y_adj), mode="array").c2c_cvg(threshold=0.10),
                "C2CCvg_33": C2CCoverage((nodes, preds), (nodes, y_adj), mode="array").c2c_cvg(threshold=0.33),
                "C2CCvg_50": C2CCoverage((nodes, preds), (nodes, y_adj), mode="array").c2c_cvg(threshold=0.50),
                "ARI": float(adjusted_rand_score(y_adj, preds)),
            }
        )

        if deps is not None and not deps.empty:
            d = deps.copy()
            if "Dependency_Count" not in d.columns:
                d["Dependency_Count"] = 1.0

            out["TurboMQ_norm"] = float(
                TurboMQ(
                    d,
                    (nodes, preds.tolist()),
                    source_col="Source_File",
                    target_col="Target_File",
                    weight_col="Dependency_Count",
                    normalized=True,
                ).score()
            )

        return out

    def run(
        self,
        data,
        Z: np.ndarray,
        clustering: str = "ahc",
        do_eval: bool = True,
        user_k: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        X = np.asarray(Z, dtype=np.float32)
        n = X.shape[0]

        nodes = self._get_nodes(data, n)

        if user_k is None:
            k_used, diag_df = self._choose_k(X, clustering)
        else:
            k_used, diag_df = int(user_k), pd.DataFrame()

        labels = self._fit_labels(X, k_used, clustering)

        row: Dict[str, Any] = {
            "Clustering_Algorithm": "KMeans" if str(clustering).lower() == "kmeans" else "AHC",
            "Recovered_clusters": int(k_used),
        }

        out: Dict[str, Any] = {
            "labels": labels.tolist(),
            "Recovered_clusters": int(k_used),
        }

        if do_eval:
            y_true = self._get_y_true(data, n)
            if y_true is None:
                raise ValueError(
                    "do_eval=True but ground-truth not found (data.y_true or data.df['Label'])."
                )
            row["GT_clusters"] = int(len(np.unique(y_true)))
            deps = self._get_deps(data)
            row.update(self._eval(labels, y_true, nodes, deps, data))

        return pd.DataFrame([row]), out
