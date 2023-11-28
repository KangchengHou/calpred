import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
from typing import List, Union, Dict
import numpy as np
import pandas as pd
import seaborn as sns

from . import logger


def plot_heatmap(
    value_df: pd.DataFrame,
    annot_df: pd.DataFrame = None,
    annot_kws: Dict = None,
    cmap="RdBu_r",
    dpi=150,
    squaresize=20,
    heatmap_linewidths=0.5,
    heatmap_linecolor="gray",
    heatmap_xticklabels=True,
    heatmap_yticklabels=True,
    heatmap_cbar=True,
    heatmap_cbar_kws=dict(use_gridspec=False, location="top", fraction=0.03, pad=0.01),
    heatmap_vmin=-5,
    heatmap_vmax=5,
    xticklabels_rotation=45,
):
    """Plot heatmap with annotations.

    Parameters
    ----------
    value_df: pd.DataFrame
        The dataframe with the values to plot.
    annot_df: pd.DataFrame
        The dataframe with the annotations to plot.
    """
    figwidth = value_df.shape[1] * squaresize / float(dpi)
    figheight = value_df.shape[0] * squaresize / float(dpi)
    fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    sns.heatmap(
        value_df,
        cmap=cmap,
        linewidths=heatmap_linewidths,
        linecolor=heatmap_linecolor,
        square=True,
        annot=annot_df,
        annot_kws=annot_kws,
        fmt="",
        ax=ax,
        xticklabels=heatmap_xticklabels,
        yticklabels=heatmap_yticklabels,
        cbar=heatmap_cbar,
        cbar_kws=heatmap_cbar_kws,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    plt.yticks(fontsize=8)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xticklabels_rotation,
        va="top",
        ha="right",
        fontsize=8,
    )
    ax.tick_params(left=False, bottom=False, pad=-2)
    trans = mtrans.Affine2D().translate(5, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)
    return fig, ax


def plot_r2_heatmap(
    value_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    baseline_df: pd.Series = None,
    cbar_pad=0.04,
    cbar_fraction=0.0188,
    squaresize=45,
    heatmap_vmin=-0.5,
    heatmap_vmax=0.5,
    heatmap_linecolor="white",
    heatmap_linewidths=1.0,
    dpi=150,
):
    """Plot heatmap of variable R2 using `plot_heatmap` functions and
    colorbar functions.

    Parameters
    ----------
    value_df : pd.DataFrame
        R2 differences.
    annot_df : pd.DataFrame
        annotations.
    cbar_pad : float, optional
        pad for colorbar, by default 0.04
    cbar_fraction : float, optional
        fraction for colorbar, by default 0.0188

    Returns
    -------
    fig, ax
    """
    value_df, pval_df = value_df.copy(), pval_df.copy()
    assert np.all(value_df.index == pval_df.index) and np.all(
        value_df.columns == pval_df.columns
    )
    annot_df = pd.DataFrame("", index=value_df.index, columns=value_df.columns)
    for r in value_df.index:
        for c in value_df.columns:
            val, pval = value_df.loc[r, c], pval_df.loc[r, c]
            if np.isnan(val):
                annot = "NA"
            elif pval < 0.05 / pval_df.size:
                annot = f"{val * 100:+.0f}%"
            elif pval < 0.05 / pval_df.shape[0]:
                annot = "*"
            else:
                annot = ""
            annot_df.loc[r, c] = annot

    # after constructing annotation, fill missing values with NA
    value_df, pval_df = value_df.fillna(0.0), pval_df.fillna(0.0)

    logger.info(f"#rows={pval_df.shape[0]}, #columns={pval_df.shape[1]}")
    logger.info(
        f"N={np.sum(pval_df.values < 0.05 / pval_df.size)} with number: p < 0.05 / {pval_df.size}; "
        f"N={np.sum(pval_df.values < 0.05 / pval_df.shape[0])} with *: p < 0.05 / {pval_df.shape[0]}"
    )
    logger.info(
        f"{np.any(pval_df <= 0.05 / pval_df.size, axis=0).sum()} PGSs with at least one significant covariate"
    )
    if baseline_df is not None:
        baseline_list = baseline_df[value_df.columns].values
        value_df.columns = [
            f"{t} ({b*100:.0f}%)" for t, b in zip(value_df.columns, baseline_list)
        ]
        annot_df.columns = [
            f"{t} ({b*100:.0f}%)" for t, b in zip(annot_df.columns, baseline_list)
        ]

    fig, ax = plot_heatmap(
        value_df=value_df,
        annot_df=annot_df,
        annot_kws={"fontsize": 6, "weight": "bold"},
        cmap=plt.get_cmap("bwr", 11),
        squaresize=squaresize,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax,
        heatmap_linecolor=heatmap_linecolor,
        heatmap_linewidths=heatmap_linewidths,
        heatmap_cbar_kws=dict(
            use_gridspec=False,
            location="right",
            fraction=cbar_fraction,
            pad=cbar_pad,
            drawedges=True,
        ),
        dpi=dpi,
    )
    ax.set_xlabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)

    ax.set_ylabel(None)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    # additional setup on colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels(["<-50%", "-25%", "0%", "25%", ">50%"])
    cbar.ax.set_ylabel(
        "Relative $\Delta (R^2)$", rotation=270, fontsize=9, labelpad=6.0
    )
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.tick_params(size=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return fig, ax


def plot_coef_heatmap(
    value_df,
    squaresize=45,
    heatmap_vmin=-0.5,
    heatmap_vmax=0.5,
    heatmap_linecolor="white",
    heatmap_linewidths=1.0,
    cbar_pad=0.04,
    cbar_fraction=0.0188,
    dpi=150,
    flip_value=False,
    cmap="bwr",
):
    if flip_value:
        value_df *= -1
    annot_df = value_df.applymap(lambda x: f"{x:.2f}" if ~np.isnan(x) else "NA")
    value_df = value_df.fillna(0.0)

    fig, ax = plot_heatmap(
        value_df=value_df,
        annot_df=annot_df,
        annot_kws={"fontsize": 6, "weight": "bold"},
        cmap=plt.get_cmap(cmap, 11),
        squaresize=squaresize,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax,
        heatmap_linecolor=heatmap_linecolor,
        heatmap_linewidths=heatmap_linewidths,
        heatmap_cbar_kws=dict(
            use_gridspec=False,
            location="right",
            fraction=cbar_fraction,
            pad=cbar_pad,
            drawedges=True,
        ),
        dpi=dpi,
    )
    ax.set_xlabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)

    ax.set_ylabel(None)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
    cbar.set_ticklabels(["-0.2", "-0.1", "0", "0.1", "0.2"])
    if flip_value:
        cbar_ylabel = r"Negative estimated $\beta$"
    else:
        cbar_ylabel = r"Estimated $\beta$"

    cbar.ax.set_ylabel(cbar_ylabel, rotation=270, fontsize=9, labelpad=12.0)

    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)

    cbar.ax.tick_params(labelsize=8)
    cbar.ax.tick_params(size=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return fig, ax


def plot_scatter_calibration(x, y, ax=None, legend=False, s=0.5, downsample=1.0):
    from scipy.stats import linregress

    x, y = np.array(x), np.array(y)
    if ax is None:
        ax = plt.gca()
    ax.axline((x.mean(), x.mean()), slope=1, ls="--", color="blue", label="y=x")
    slope, intercept = linregress(x=x, y=y)[0:2]

    if downsample < 1:
        n = len(x)
        idx = np.random.choice(np.arange(n), size=int(n * downsample), replace=False)
        x, y = x[idx], y[idx]
    ax.scatter(x, y, s=s, marker=".", edgecolors="C0", c="C0")

    ax.axline(
        (0, intercept),
        slope=slope,
        ls="--",
        color="black",
        lw=1,
        label=f"y={slope:.2f}x+{intercept:.2f}",
    )

    if legend:
        ax.legend(loc="upper left", fontsize=8)


def plot_prob_calibration(prob, y, n_q=30, ax=None, color="blue", label=None, ci=1.96):
    if ax is None:
        ax = plt.gca()

    df = pd.DataFrame({"prob": prob, "y": y})
    df["q"] = pd.qcut(df["prob"], q=n_q).cat.codes + 1

    stats_df = []
    for q, qdf in df.groupby("q"):
        stats_df.append(
            [qdf["prob"].mean(), qdf["y"].mean(), qdf["y"].std() / np.sqrt(len(qdf))]
        )
    stats_df = pd.DataFrame(stats_df, columns=["prob", "y", "y_std"])

    ax.errorbar(
        stats_df["prob"],
        stats_df["y"],
        yerr=ci * stats_df["y_std"],
        fmt=".",
        markersize=2,
        color=color,
        elinewidth=0.5,
        capsize=2,
        label=label,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    ax.plot(stats_df["prob"], stats_df["y"], lw=0.5, color=color)
    ax.axline((0, 0), slope=1, ls="-", color="red", lw=1)


# def plot_intervals(idx, ax=None):
#     if ax is None:
#         ax = plt.gca()
#     ax.scatter(q2x(model.fittedvalues), data_df["VitD"], s=1, color="black")
#     center = data_df["VitD"].mean()
#     ax.axline((center, center), slope=1, color="red", ls="--")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Phenotype")

#     # plot top and bottom individuals
#     y = q2x(df.loc[idx, "mean"])
#     low = q2x(df.loc[idx, "mean"] - df.loc[idx, "sd"] * 1.645)
#     high = q2x(df.loc[idx, "mean"] + df.loc[idx, "sd"] * 1.645)

#     ax.errorbar(
#         x=y,
#         y=y,
#         yerr=[y - low, high - y],
#         fmt=".",
#         color="red",
#         capsize=3,
#         linewidth=1.0,
#     )
#     # annotate bottom
#     ax.text(
#         x=y[0],
#         y=high[0] + 5,
#         s=f"[{low[0]:.1f}, {high[0]:.1f}]",
#         color="red",
#         ha="center",
#         va="bottom",
#         fontsize=15,
#     )

#     # annotate top
#     ax.text(
#         x=y[-1],
#         y=high[-1] + 5,
#         s=f"[{low[-1]:.1f}, {high[-1]:.1f}]",
#         color="red",
#         ha="center",
#         va="bottom",
#         fontsize=15,
#     )
#     ax.axhline(y=25, ls="--", color="blue", lw=0.5)
#     ax.axhline(y=80, ls="--", color="blue", lw=0.5)
#     ax.axhspan(ymin=25, ymax=80, color="blue", alpha=0.1)
#     ax.text(
#         x=q2x(np.mean(model.fittedvalues)),
#         y=80,
#         ha="center",
#         va="top",
#         s="European RI",
#         fontsize=15,
#         color="blue",
#     )

#     return fig, ax
