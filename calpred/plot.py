import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
from typing import List, Union, Dict
import numpy as np
import pandas as pd
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