"""
Benchmark EKM initialization strategies:
  1) Internal `_kmeans_plus_init` (via EKM(init='plus'))
  2) sklearn.cluster.kmeans_plusplus (passed as explicit centers)

比较 EKM 使用内部 `_kmeans_plus_init` 与 sklearn `kmeans_plusplus` 作为初始化后得到的聚类结果质量与稳定性，帮助选择更准确、鲁棒的初始化方式。

核心指标：
  * ekm_objective: EKM 内部的近似目标 (脚本用最终 obj: model.objective_)
  * final_SSE: 最终簇中心下样本到最近中心的平方距离和（惯性）
  * init_SSE: 仅用初始化中心计算的 SSE（未迭代前的质量）
  * iterations: EKM 收敛所需迭代次数
  * runtime_fit: 拟合耗时 (s)
  * (可选) ARI / NMI / ACC：若提供标签
  * Robustness: 统计 (mean / std / coeff var) 以及两个初始化之间胜出次数

命令行示例 (Windows cmd)：
  cd python
  python benchmark_ekm_init_compare.py --n-samples 4000 --n-features 10 --n-clusters 12 --n-runs 50 --seed 42 --verbose

使用已有数据 (CSV / NPY)：
  python benchmark_ekm_init_compare.py --data-path ..\reproduction\heart\heart.csv --label-col -1 --n-clusters 4 --n-runs 30

若标签列为最后一列，指定 --label-col -1 即可。若数据无标签，不要加参数。

可选计算 silhouette（较耗时 O(N^2)）：
  python benchmark_ekm_init_compare.py --compute-silhouette --silhouette-subsample 5000

输出：
  * 每次 run 的详细行 (verbose 模式)
  * 汇总统计
  * 可选 CSV (--save-csv) 保存逐次结果

注意：
  * sklearn 的 kmeans_plusplus 仅支持欧式距离；脚本限制 metric=euclidean 比较初始化。
  * 如果需要比较曼哈顿距离，可后续扩展内部实现，但 sklearn 版本不可直接支持。
  * ACC 使用匈牙利算法；若未安装 SciPy，则回退为贪心近似 (提示)。

Author: Auto-generated helper script.
"""
from __future__ import annotations
import argparse
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import warnings

from sklearn.cluster import kmeans_plusplus  # type: ignore
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _SCIPY_AVAILABLE = False

from ekm_sklearn import EKM, _kmeans_plus_init  # type: ignore
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

try:
    import matplotlib.pyplot as plt  # type: ignore
    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    _MPL_AVAILABLE = False

# ---------------- distance helpers ---------------- #

def pairwise_distances_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return a2 + b2 - 2 * a @ b.T

# ---------------- metrics ---------------- #

def sse(X: np.ndarray, centers: np.ndarray) -> float:
    D2 = pairwise_distances_squared(X, centers)
    return float(np.min(D2, axis=1).sum())

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Permutation-invariant accuracy via Hungarian (if SciPy). Greedy fallback otherwise."""
    cm = confusion_matrix(y_true, y_pred)
    if _SCIPY_AVAILABLE:
        # Hungarian solves min-cost; convert to cost matrix
        cost = cm.max() - cm
        r, c = linear_sum_assignment(cost)
        return float(cm[r, c].sum() / y_true.size)
    # greedy fallback
    cm_copy = cm.copy()
    acc = 0
    used_cols = set()
    for _ in range(min(cm.shape)):
        best = (-1, -1, -1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if j in used_cols:
                    continue
                if cm_copy[i, j] > best[2]:
                    best = (i, j, cm_copy[i, j])
        if best[1] >= 0:
            used_cols.add(best[1])
            acc += best[2]
    return float(acc / y_true.size)

# ---------------- data loading / generation ---------------- #

def load_matrix_with_labels(path: str, label_col: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == '.npy':
        M = np.load(p)
    else:
        # try comma, else whitespace
        try:
            M = np.loadtxt(p, delimiter=',')
        except Exception:
            M = np.loadtxt(p)
    if M.ndim != 2:
        raise ValueError("Loaded data must be 2D array")
    if label_col is None:
        return M.astype(float), None
    # interpret negative index
    if label_col < 0:
        label_col = M.shape[1] + label_col
    if label_col < 0 or label_col >= M.shape[1]:
        raise ValueError("label_col out of range")
    y = M[:, label_col].astype(int)
    X = np.delete(M, label_col, axis=1).astype(float)
    return X, y

def gen_dirichlet_synthetic(n_samples: int, n_features: int, k_true: int, seed: int, imbalance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Previous (备用) 合成方式: Dirichlet 不均衡多高维高斯混合。"""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-10, 10, size=(k_true, n_features))
    alpha = np.ones(k_true)
    alpha[0] = imbalance
    probs = rng.dirichlet(alpha)
    counts = rng.multinomial(n_samples, probs)
    X_list, y_list = [], []
    for k in range(k_true):
        if counts[k] == 0:
            continue
        scale = rng.uniform(0.3, 2.0)
        cov = np.eye(n_features) * (scale ** 2)
        Xk = rng.multivariate_normal(centers[k], cov, size=counts[k])
        X_list.append(Xk)
        y_list.append(np.full(counts[k], k, int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]

def gen_imbalanced_benchmark(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用 benchmark.py 中固定的不平衡 2D blobs 数据集.

    配置：
        样本数: [2000, 50, 30]
        中心:   [(-5, -2), (0, 0), (5, 5)]
        标准差: [1.0, 1.0, 1.0]
    返回 (X, y)
    """
    X, y = make_blobs(
        n_samples=[2000, 50, 30],
        centers=[(-5, -2), (0, 0), (5, 5)],
        cluster_std=[1.0, 1.0, 1.0],
        random_state=seed
    )
    return X.astype(float), y.astype(int)

# ---------------- run container ---------------- #

@dataclass
class RunRecord:
    seed: int
    init_sse_internal: float
    init_sse_sklearn: float
    final_sse_internal: float
    final_sse_sklearn: float
    obj_internal: float
    obj_sklearn: float
    iters_internal: int
    iters_sklearn: int
    time_internal: float
    time_sklearn: float
    ari_internal: Optional[float]
    ari_sklearn: Optional[float]
    nmi_internal: Optional[float]
    nmi_sklearn: Optional[float]
    acc_internal: Optional[float]
    acc_sklearn: Optional[float]
    sil_internal: Optional[float]
    sil_sklearn: Optional[float]

# ---------------- core benchmark ---------------- #

def run_once(X: np.ndarray, K: int, seed: int, y: Optional[np.ndarray], compute_sil: bool, sil_subsample: int,
             return_models: bool = False):
    # initialization centers
    np.random.seed(seed)
    centers_int = _kmeans_plus_init(X, K, metric='euclidean', random_state=seed)
    init_sse_int = sse(X, centers_int)

    centers_sk, _ = kmeans_plusplus(X, n_clusters=K, random_state=seed)
    init_sse_sk = sse(X, centers_sk)

    # Fit internal version
    t0 = time.perf_counter()
    model_int = EKM(n_clusters=K, init=centers_int.copy(), random_state=seed, n_init=1, metric='euclidean')
    model_int.fit(X)
    t1 = time.perf_counter()

    # Fit sklearn-initialized version by passing centers directly
    t2 = time.perf_counter()
    model_sk = EKM(n_clusters=K, init=centers_sk.copy(), random_state=seed, n_init=1, metric='euclidean')
    model_sk.fit(X)
    t3 = time.perf_counter()

    final_sse_int = sse(X, model_int.cluster_centers_)
    final_sse_sk = sse(X, model_sk.cluster_centers_)

    ari_int = ari_sk = nmi_int = nmi_sk = acc_int = acc_sk = None
    if y is not None:
        ari_int = adjusted_rand_score(y, model_int.labels_)
        ari_sk = adjusted_rand_score(y, model_sk.labels_)
        nmi_int = normalized_mutual_info_score(y, model_int.labels_)
        nmi_sk = normalized_mutual_info_score(y, model_sk.labels_)
        acc_int = cluster_accuracy(y, model_int.labels_)
        acc_sk = cluster_accuracy(y, model_sk.labels_)

    sil_int = sil_sk = None
    if compute_sil:
        if X.shape[0] > sil_subsample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=sil_subsample, replace=False)
            Xs = X[idx]
            labels_int = model_int.labels_[idx]
            labels_sk = model_sk.labels_[idx]
        else:
            Xs = X
            labels_int = model_int.labels_
            labels_sk = model_sk.labels_
        # silhouette requires at least 2 clusters with >1 sample
        def safe_sil(data, labels):
            try:
                if len(np.unique(labels)) <= 1:
                    return None
                return float(silhouette_score(data, labels, metric='euclidean'))
            except Exception:
                return None
        sil_int = safe_sil(Xs, labels_int)
        sil_sk = safe_sil(Xs, labels_sk)

    record = RunRecord(
        seed=seed,
        init_sse_internal=init_sse_int,
        init_sse_sklearn=init_sse_sk,
        final_sse_internal=final_sse_int,
        final_sse_sklearn=final_sse_sk,
        obj_internal=model_int.objective_,
        obj_sklearn=model_sk.objective_,
        iters_internal=model_int.n_iter_,
        iters_sklearn=model_sk.n_iter_,
        time_internal=t1 - t0,
        time_sklearn=t3 - t2,
        ari_internal=ari_int,
        ari_sklearn=ari_sk,
        nmi_internal=nmi_int,
        nmi_sklearn=nmi_sk,
        acc_internal=acc_int,
        acc_sklearn=acc_sk,
        sil_internal=sil_int,
        sil_sklearn=sil_sk,
    )
    if return_models:
        return record, model_int, model_sk, centers_int, centers_sk
    return record

# ---------------- aggregation ---------------- #

def mean_std(arr: List[float]) -> Tuple[float, float]:
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan')
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

def coef_var(arr: List[float]) -> float:
    m, s = mean_std(arr)
    if not np.isfinite(m) or m == 0:
        return float('nan')
    return float(s / m)

# ---------------- main loop ---------------- #

def _project_2d(X: np.ndarray, centers_a: np.ndarray, centers_b: np.ndarray):
    """Return (X2d, C_a2d, C_b2d, projector) using identity if 2D else PCA(2)."""
    if X.shape[1] <= 2:
        return X[:, :2], centers_a[:, :2], centers_b[:, :2], None
    pca = PCA(n_components=2, random_state=0)
    X2d = pca.fit_transform(X)
    Ca2d = pca.transform(centers_a)
    Cb2d = pca.transform(centers_b)
    return X2d, Ca2d, Cb2d, pca

def _plot_clusters(X: np.ndarray, y_true: Optional[np.ndarray], model_int: EKM, model_sk: EKM,
                   init_int: np.ndarray, init_sk: np.ndarray, args):
    if not _MPL_AVAILABLE:
        warnings.warn("matplotlib 未安装，无法绘图。请 pip install matplotlib")
        return
    # project
    X2d, Cint2d, Csk2d, _ = _project_2d(X, model_int.cluster_centers_, model_sk.cluster_centers_)
    if args.plot_show_initial:
        _, Iint2d, Isk2d, _ = _project_2d(X, init_int, init_sk)
    # figure with 3 panels: data+int, data+sk, overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cm = plt.get_cmap('tab10')
    # internal
    ax = axes[0]
    if y_true is not None:
        for lab in np.unique(y_true):
            pts = X2d[y_true == lab]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, color=cm(lab % 10), label=f'y={lab}')
    else:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=6, alpha=0.5, color='#888888')
    ax.scatter(Cint2d[:, 0], Cint2d[:, 1], c='red', marker='X', s=120, edgecolor='k', label='EKM-centers (int init)')
    if args.plot_show_initial:
        ax.scatter(Iint2d[:, 0], Iint2d[:, 1], c='none', edgecolor='red', marker='o', s=80, label='init centers')
    ax.set_title('Internal init result')
    ax.legend(fontsize=8, loc='best')
    # sklearn init
    ax = axes[1]
    if y_true is not None:
        for lab in np.unique(y_true):
            pts = X2d[y_true == lab]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, color=cm(lab % 10))
    else:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=6, alpha=0.5, color='#888888')
    ax.scatter(Csk2d[:, 0], Csk2d[:, 1], c='blue', marker='D', s=120, edgecolor='k', label='EKM-centers (sk init)')
    if args.plot_show_initial:
        ax.scatter(Isk2d[:, 0], Isk2d[:, 1], c='none', edgecolor='blue', marker='o', s=80, label='init centers')
    ax.set_title('sklearn init result')
    ax.legend(fontsize=8, loc='best')
    # overlay
    ax = axes[2]
    ax.scatter(Cint2d[:, 0], Cint2d[:, 1], c='red', marker='X', s=140, edgecolor='k', label='centers-int')
    ax.scatter(Csk2d[:, 0], Csk2d[:, 1], c='blue', marker='D', s=140, edgecolor='k', label='centers-sk')
    if args.plot_show_initial:
        ax.scatter(Iint2d[:, 0], Iint2d[:, 1], c='none', edgecolor='red', marker='o', s=90, label='init-int')
        ax.scatter(Isk2d[:, 0], Isk2d[:, 1], c='none', edgecolor='blue', marker='o', s=90, label='init-sk')
    ax.set_title('Center overlay')
    ax.legend(fontsize=8, loc='best')
    for a in axes:
        a.set_xlabel('Dim 1')
        a.set_ylabel('Dim 2')
    fig.suptitle('EKM Initialization Comparison', fontsize=14)
    fig.tight_layout()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=200)
        print(f"[Plot] 保存图像到 {args.save_plot}")
    if args.plot:
        plt.show()

def benchmark(args):
    if args.data_path:
        X, y = load_matrix_with_labels(args.data_path, args.label_col)
    else:
        if args.synthetic_mode == 'benchmark':
            # 固定 3 类不平衡数据集
            X, y = gen_imbalanced_benchmark(args.seed)
            # 强制覆盖 n_clusters=3 以避免不匹配
            if args.n_clusters != 3:
                print(f"[Info] 覆盖 --n-clusters={args.n_clusters} 为 3 (benchmark 模式固定三类)")
                args.n_clusters = 3
        else:
            X, y = gen_dirichlet_synthetic(args.n_samples, args.n_features, max(args.n_clusters, 2), args.seed, args.imbalance)
    N, P = X.shape
    K = args.n_clusters
    if K >= N:
        raise ValueError("n_clusters must be < n_samples")

    seeds = [args.seed + i for i in range(args.n_runs)]
    records: List[RunRecord] = []

    cached_plot_tuple = None
    for s in seeds:
        need_models = args.plot and (args.plot_seed is not None) and (s == args.plot_seed)
        if need_models:
            rec, m_int, m_sk, init_int, init_sk = run_once(X, K, s, y, args.compute_silhouette, args.silhouette_subsample, return_models=True)
            cached_plot_tuple = (m_int, m_sk, init_int, init_sk)
        else:
            rec = run_once(X, K, s, y, args.compute_silhouette, args.silhouette_subsample)
        records.append(rec)
        if args.verbose:
            line = (f"seed {s:4d} | initSSE int {rec.init_sse_internal:10.2f} sk {rec.init_sse_sklearn:10.2f} | "
                    f"finalSSE int {rec.final_sse_internal:10.2f} sk {rec.final_sse_sklearn:10.2f} | "
                    f"obj int {rec.obj_internal:10.2f} sk {rec.obj_sklearn:10.2f} | "
                    f"iters int {rec.iters_internal:3d} sk {rec.iters_sklearn:3d} | "
                    f"time int {rec.time_internal*1e3:7.2f}ms sk {rec.time_sklearn*1e3:7.2f}ms")
            if y is not None:
                line += f" | ARI int {rec.ari_internal:.4f} sk {rec.ari_sklearn:.4f} | NMI int {rec.nmi_internal:.4f} sk {rec.nmi_sklearn:.4f}"
            print(line)

    # aggregate helper
    def collect(name: str) -> List[float]:
        return [getattr(r, name) for r in records if getattr(r, name) is not None]

    def summarize_pair(a_name: str, b_name: str, label: str):
        a = collect(a_name); b = collect(b_name)
        ma, sa = mean_std(a); mb, sb = mean_std(b)
        diff = [x - y for x, y in zip(a, b)]
        md, sd = mean_std(diff)
        wins_a = sum(1 for x, y in zip(a, b) if x < y)  # smaller is better
        wins_b = sum(1 for x, y in zip(a, b) if y < x)
        return {
            f'{label}_internal_mean': ma,
            f'{label}_internal_std': sa,
            f'{label}_sklearn_mean': mb,
            f'{label}_sklearn_std': sb,
            f'{label}_diff_mean': md,
            f'{label}_diff_std': sd,
            f'{label}_wins_internal': wins_a,
            f'{label}_wins_sklearn': wins_b,
        }

    summary: Dict[str, float] = {}
    # Metrics where smaller is better
    for lab, a_name, b_name in [
        ('initSSE', 'init_sse_internal', 'init_sse_sklearn'),
        ('finalSSE', 'final_sse_internal', 'final_sse_sklearn'),
        ('objective', 'obj_internal', 'obj_sklearn'),
        ('runtime', 'time_internal', 'time_sklearn'),
        ('iterations', 'iters_internal', 'iters_sklearn'),
    ]:
        summary.update(summarize_pair(a_name, b_name, lab))

    # Variances for robustness (final SSE & objective)
    final_sse_int = collect('final_sse_internal')
    final_sse_sk = collect('final_sse_sklearn')
    obj_int = collect('obj_internal')
    obj_sk = collect('obj_sklearn')
    summary['finalSSE_internal_cv'] = coef_var(final_sse_int)
    summary['finalSSE_sklearn_cv'] = coef_var(final_sse_sk)
    summary['objective_internal_cv'] = coef_var(obj_int)
    summary['objective_sklearn_cv'] = coef_var(obj_sk)

    # External metrics (larger is better) -> adapt pair summarization
    if y is not None:
        for metric_label, a_name, b_name in [
            ('ARI', 'ari_internal', 'ari_sklearn'),
            ('NMI', 'nmi_internal', 'nmi_sklearn'),
            ('ACC', 'acc_internal', 'acc_sklearn'),
        ]:
            a = collect(a_name); b = collect(b_name)
            ma, sa = mean_std(a); mb, sb = mean_std(b)
            diff = [x - y for x, y in zip(a, b)]
            md, sd = mean_std(diff)
            wins_a = sum(1 for x, y in zip(a, b) if x > y)  # larger better
            wins_b = sum(1 for x, y in zip(a, b) if y > x)
            summary.update({
                f'{metric_label}_internal_mean': ma,
                f'{metric_label}_internal_std': sa,
                f'{metric_label}_sklearn_mean': mb,
                f'{metric_label}_sklearn_std': sb,
                f'{metric_label}_diff_mean': md,
                f'{metric_label}_diff_std': sd,
                f'{metric_label}_wins_internal': wins_a,
                f'{metric_label}_wins_sklearn': wins_b,
            })

    if args.compute_silhouette:
        for metric_label, a_name, b_name in [
            ('SIL', 'sil_internal', 'sil_sklearn'),
        ]:
            a = collect(a_name); b = collect(b_name)
            if a and b:
                ma, sa = mean_std(a); mb, sb = mean_std(b)
                diff = [x - y for x, y in zip(a, b)]
                md, sd = mean_std(diff)
                wins_a = sum(1 for x, y in zip(a, b) if x > y)
                wins_b = sum(1 for x, y in zip(a, b) if y > x)
                summary.update({
                    f'{metric_label}_internal_mean': ma,
                    f'{metric_label}_internal_std': sa,
                    f'{metric_label}_sklearn_mean': mb,
                    f'{metric_label}_sklearn_std': sb,
                    f'{metric_label}_diff_mean': md,
                    f'{metric_label}_diff_std': sd,
                    f'{metric_label}_wins_internal': wins_a,
                    f'{metric_label}_wins_sklearn': wins_b,
                })

    # ---- print summary ----
    print("\n================ SUMMARY ================")
    print(f"Runs={len(records)} | n_samples={N} | n_features={P} | K={K}")
    def pr(label: str, better: str):
        m_int = summary.get(f'{label}_internal_mean', float('nan'))
        s_int = summary.get(f'{label}_internal_std', float('nan'))
        m_sk = summary.get(f'{label}_sklearn_mean', float('nan'))
        s_sk = summary.get(f'{label}_sklearn_std', float('nan'))
        wins_int = summary.get(f'{label}_wins_internal', 0)
        wins_sk = summary.get(f'{label}_wins_sklearn', 0)
        print(f"{label:<12} int {m_int:10.4f}±{s_int:<10.4f} | sk {m_sk:10.4f}±{s_sk:<10.4f} | wins int/sk {wins_int}/{wins_sk} ({better} better)")
    # smaller is better
    for l in ['initSSE','finalSSE','objective','iterations','runtime']:
        pr(l, 'smaller')
    # robustness
    print(f"CV finalSSE int {summary['finalSSE_internal_cv']:.4f} | sk {summary['finalSSE_sklearn_cv']:.4f}")
    print(f"CV objective int {summary['objective_internal_cv']:.4f} | sk {summary['objective_sklearn_cv']:.4f}")
    if y is not None:
        for l in ['ARI','NMI','ACC']:
            pr(l, 'larger')
    if args.compute_silhouette:
        pr('SIL', 'larger')

    # optional CSV export
    if args.save_csv:
        out = Path(args.save_csv)
        import csv
        with out.open('w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'seed','init_sse_internal','init_sse_sklearn','final_sse_internal','final_sse_sklearn',
                'obj_internal','obj_sklearn','iters_internal','iters_sklearn','time_internal','time_sklearn',
                'ari_internal','ari_sklearn','nmi_internal','nmi_sklearn','acc_internal','acc_sklearn',
                'sil_internal','sil_sklearn'
            ]
            writer.writerow(header)
            for r in records:
                writer.writerow([
                    r.seed,r.init_sse_internal,r.init_sse_sklearn,r.final_sse_internal,r.final_sse_sklearn,
                    r.obj_internal,r.obj_sklearn,r.iters_internal,r.iters_sklearn,r.time_internal,r.time_sklearn,
                    r.ari_internal,r.ari_sklearn,r.nmi_internal,r.nmi_sklearn,r.acc_internal,r.acc_sklearn,
                    r.sil_internal,r.sil_sklearn
                ])
        print(f"Per-run detailed results saved to {out.resolve()}")

    # plot after summary to not block text output
    if args.plot:
        if args.plot_seed is None:
            # rerun first seed with model capture if not already captured
            seed_plot = seeds[0]
            print(f"[Plot] 未指定 --plot-seed，使用第一个种子 {seed_plot}")
            rec2, m_int, m_sk, init_int, init_sk = run_once(X, K, seed_plot, y, args.compute_silhouette, args.silhouette_subsample, return_models=True)
            _plot_clusters(X, y, m_int, m_sk, init_int, init_sk, args)
        else:
            if cached_plot_tuple is None:
                print(f"[Warning] 未找到指定 plot seed {args.plot_seed} 的缓存，重新运行获取模型。")
                rec2, m_int, m_sk, init_int, init_sk = run_once(X, K, args.plot_seed, y, args.compute_silhouette, args.silhouette_subsample, return_models=True)
                _plot_clusters(X, y, m_int, m_sk, init_int, init_sk, args)
            else:
                m_int, m_sk, init_int, init_sk = cached_plot_tuple
                _plot_clusters(X, y, m_int, m_sk, init_int, init_sk, args)

    return summary

# ---------------- CLI ---------------- #

def parse_args():
    ap = argparse.ArgumentParser(description='Compare EKM initialization: internal vs sklearn kmeans++')
    ap.add_argument('--data-path', type=str, default='', help='Optional data file (CSV/NPY). If given, synthetic generation is skipped.')
    ap.add_argument('--label-col', type=int, default=None, help='Column index of labels in data (e.g., -1 for last). If omitted, treat as unlabeled.')
    ap.add_argument('--synthetic-mode', type=str, default='benchmark', choices=['benchmark','dirichlet'],
                   help='benchmark: 使用 benchmark.py 固定不平衡数据; dirichlet: 旧版可变维度不平衡合成')
    ap.add_argument('--n-samples', type=int, default=3000, help='(dirichlet 模式) 样本数')
    ap.add_argument('--n-features', type=int, default=8, help='(dirichlet 模式) 特征数')
    ap.add_argument('--n-clusters', type=int, default=10, help='目标聚类数 (benchmark 模式会被强制为 3)')
    ap.add_argument('--n-runs', type=int, default=30, help='随机种子 / 重复次数')
    ap.add_argument('--seed', type=int, default=42, help='基础随机种子')
    ap.add_argument('--imbalance', type=float, default=8.0, help='(dirichlet 模式) 不平衡强度')
    ap.add_argument('--compute-silhouette', action='store_true', help='计算 silhouette (O(N^2)，可能较慢)')
    ap.add_argument('--silhouette-subsample', type=int, default=4000, help='silhouette 计算的子样本上限')
    ap.add_argument('--save-csv', type=str, default='', help='保存逐次结果到 CSV')
    ap.add_argument('--verbose', action='store_true', help='打印每次 run 的详细指标')
    ap.add_argument('--plot', action='store_true', help='显示聚类结果与中心对比图 (2D 或 PCA 降维)')
    ap.add_argument('--plot-seed', type=int, default=None, help='指定哪一个种子的结果绘图 (默认首个)')
    ap.add_argument('--save-plot', type=str, default='', help='保存图像到文件 (例如 plot.png)')
    ap.add_argument('--plot-show-initial', action='store_true', help='在图中同时显示初始化中心 (圈)')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    benchmark(args)
