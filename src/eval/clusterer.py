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
    def __init__(self, k_range: Iterable[int] = range(10, 31), sample_size: int = 2000):
        self.k_range = list(k_range)
        self.sample_size = int(sample_size)
       

    def _get_nodes(self, data, n: int) -> List[str]:
        if hasattr(data, "nodes") and data.nodes is not None and len(data.nodes) == n:
            return [str(x) for x in list(data.nodes)]
        if hasattr(data, "node_list") and data.node_list is not None and len(data.node_list) == n:
            return [str(x) for x in list(data.node_list)]
        if hasattr(data, "df") and isinstance(data.df, pd.DataFrame) and "File" in data.df.columns and len(data.df) == n:
            return data.df["File"].astype(str).tolist()
        return [str(i) for i in range(n)]

    def _get_y_true(self, data, n: int) -> Optional[np.ndarray]:
        if hasattr(data, "y_true") and data.y_true is not None and len(data.y_true) == n:
            return np.asarray(data.y_true, dtype=int)
        if hasattr(data, "df") and isinstance(data.df, pd.DataFrame) and "Label" in data.df.columns and len(data.df) == n:
            return data.df["Label"].to_numpy(dtype=int)
        return None


    def _get_deps(self, data) -> Optional[pd.DataFrame]:
        if hasattr(data, "df_dep") and isinstance(data.df_dep, pd.DataFrame):
            return data.df_dep
        return None

    def _sanitize_k(self, n: int) -> List[int]:
        ks = sorted({int(k) for k in self.k_range if 1 < int(k) < n})
        if not ks:
            raise ValueError(f"No valid k in k_range for n={n}")
        return ks

    def _fit_labels(self, X: np.ndarray, k: int, clustering: str) -> np.ndarray:
        k = int(k)
        if clustering == "kmeans":
            return KMeans(n_clusters=k).fit_predict(X)
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)

    def _best_k_elbow(self, X: np.ndarray) -> Tuple[int, pd.DataFrame]:
        n = X.shape[0]
        ks = self._sanitize_k(n)

        m = min(self.sample_size, n)
        idx = np.random.choice(n, size=m, replace=False)
        Xs = X[idx]

        inertias: List[float] = []
        for k in ks:
            km = KMeans(n_clusters=int(k)).fit(Xs)
            inertias.append(float(km.inertia_))

        if len(ks) <= 2:
            df = pd.DataFrame({"k": ks, "inertia": inertias})
            return int(ks[0]), df

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
        df = pd.DataFrame({"k": ks, "inertia": inertias, "elbow_dist": d.tolist()})
        return best_k, df

    def _eval(self, labels: np.ndarray, y_true: np.ndarray, nodes: List[str], deps: Optional[pd.DataFrame]) -> Dict[str, float]:
        preds, _ = pd.factorize(labels)
        y = np.asarray(y_true)

        out = {
            "MoJoFM": MoJoCalculator(preds, y, mode="array").mojofm(),
            "A2A": A2ACalculator(preds, y, mode="array").a2a(),
            "C2CCvg_10": C2CCoverage((nodes, preds), (nodes, y), mode="array").c2c_cvg(threshold=0.10),
            "C2CCvg_33": C2CCoverage((nodes, preds), (nodes, y), mode="array").c2c_cvg(threshold=0.33),
            "C2CCvg_50": C2CCoverage((nodes, preds), (nodes, y), mode="array").c2c_cvg(threshold=0.50),
            "ARI": float(adjusted_rand_score(y, preds)),
        }

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
            k_used, diag_df = self._best_k_elbow(X)
        else:
            k_used, diag_df = int(user_k), pd.DataFrame()

        labels = self._fit_labels(X, k_used, clustering)

        row: Dict[str, Any] = {
            "Clustering": "KMeans" if clustering == "kmeans" else "Agglomerative",
            "k_used": int(k_used),
            "Clusters": int(len(np.unique(labels))),
        }

        out: Dict[str, Any] = {"labels": labels.tolist(), "k_used": int(k_used), "k_search": diag_df}

        if do_eval:
            y_true = self._get_y_true(data, n)
            if y_true is None:
                raise ValueError("do_eval=True but ground-truth not found (data.y_true or data.df['Label']).")
            deps = self._get_deps(data)
            row.update(self._eval(labels, y_true, nodes, deps))

        return pd.DataFrame([row]), out
