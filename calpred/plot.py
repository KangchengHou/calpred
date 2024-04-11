import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
from typing import List, Union, Dict
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
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
    annot_df: pd.DataFrame = None,
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
    if annot_df is None:
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
    cbar.ax.set_ylabel("Relative $\Delta R^2$", rotation=270, fontsize=9, labelpad=6.0)
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
    cbar_ticks=[-0.2, -0.1, 0, 0.1, 0.2],
):
    if flip_value:
        value_df *= -1
    annot_df = value_df.map(lambda x: f"{x:.2f}" if ~np.isnan(x) else "NA")
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
    cbar.set_ticks(cbar_ticks)
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


def plot_scatter_calibration(x, y, ax=None, legend=False, s=5, downsample=1.0):
    from scipy.stats import linregress

    x, y = np.array(x), np.array(y)
    if ax is None:
        ax = plt.gca()
    ax.axline(
        (x.mean(), x.mean()),
        slope=1,
        ls="--",
        color="black",
        label="y=x",
        lw=1,
    )
    slope, intercept = linregress(x=x, y=y)[0:2]

    if downsample < 1:
        n = len(x)
        idx = np.random.choice(np.arange(n), size=int(n * downsample), replace=False)
        x, y = x[idx], y[idx]
    ax.scatter(x, y, s=s, marker=".", edgecolors="none", c="C0")

    ax.axline(
        (0, intercept),
        slope=slope,
        ls="--",
        color="red",
        # label=f"y={slope:.2f}x+{intercept:.2f}",
        label=f"y={slope:.2f}x",
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
    ax.axline((0, 0), slope=1, ls="--", color="black", lw=0.5)


def compare_values(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = None,
    ylabel: str = None,
    ax=None,
    s: int = 5,
):
    """Compare two p-values.
    Parameters
    ----------
    x_pval: np.ndarray
        The p-value for the first variable.
    y_pval: np.ndarray
        The p-value for the second variable.
    xlabel: str
        The label for the first variable.
    ylabel: str
        The label for the second variable.
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    nonnan_idx = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[nonnan_idx], y[nonnan_idx]
    ax.scatter(x, y, s=s)
    lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.1
    ax.axline((0, 0), slope=1, color="k", ls="--", alpha=0.5, lw=1, label="y=x")

    # add a regression line
    slope = np.linalg.lstsq(x[:, None], y[:, None], rcond=None)[0].item()

    ax.axline(
        (0, 0),
        slope=slope,
        color="black",
        ls="--",
        lw=1,
        label=f"y={slope:.2f} x",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def lighten_color(color, amount=1.25):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def color_boxplot(bplot, color):
    for i in range(len(bplot["boxes"])):
        for obj in ["whiskers", "caps", "fliers", "medians"]:
            for patch in bplot[obj]:
                patch.set_color(color)
        bplot["boxes"][i].set(color=color)


def _group_plot(
    df,
    val_col,
    groups,
    axes,
    pos_offset,
    color,
    plot_type="box",
    edge_alpha=None,
    widths=0.2,
):
    """Box / line plots for each group (in each panel)
    df should contain "group", "subgroup"
    each group corresponds to a panel, each subgroup corresponds to
    different x within the panel

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing 'group', 'subgroup', val_col
    val_col : str
        column containing the values
    """
    assert plot_type in ["box", "line"]

    for group_i, group in enumerate(groups):
        df_group = df[df.group == group]
        dict_val = {
            group: df_tmp[val_col].values
            for group, df_tmp in df_group.groupby("subgroup")
        }
        x = list(dict_val.keys())
        vals = list(dict_val.values())
        means = [np.mean(_) for _ in vals]
        sems = [np.std(_) / np.sqrt(len(_)) for _ in vals]
        if plot_type == "box":
            props = {"linewidth": 0.65}
            bplot = axes[group_i].boxplot(
                positions=np.arange(len(vals)) + pos_offset,
                x=vals,
                sym="",
                widths=widths,
                patch_artist=True,
                boxprops=props,
                whiskerprops=props,
                capprops=props,
                medianprops=props,
            )
            if edge_alpha is not None:
                color_boxplot(bplot, lighten_color(color, edge_alpha))
            else:
                for patch in bplot["medians"]:
                    patch.set_color("black")

            for patch in bplot["boxes"]:
                patch.set_facecolor(color)

        elif plot_type == "line":
            axes[group_i].errorbar(
                x=np.arange(len(vals)) + 1 + pos_offset,
                y=means,
                yerr=sems,
                fmt=".--",
                ms=4,
                mew=1,
                linewidth=1,
                color=color,
            )
        else:
            raise ValueError("plot_type must be 'box' or 'line'")

        axes[group_i].set_xlabel(group)
        axes[group_i].set_xticks(np.arange(len(vals)))
        axes[group_i].set_xticklabels(x)


def plot_group_r2(
    df: pd.DataFrame,
    figsize=(7, 1.5),
    groups=None,
    width_ratios=None,
    plot_type="box",
    color="lightgray",
):
    """Plot R2 by groups

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing 'group', 'subgroup', 'r2'
    figsize : tuple, optional
        figure size, by default (7, 1.5)
    """

    if groups is None:
        groups = df["group"].unique()
    if width_ratios is None:
        width_ratios = (
            np.array([len(df[df["group"] == g]["subgroup"].unique()) for g in groups])
            + 1
        )

    fig, axes = plt.subplots(
        figsize=figsize,
        dpi=150,
        ncols=len(width_ratios),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )

    for i, group in enumerate(groups):
        if plot_type == "bar":
            r2 = df[df["group"] == group].groupby("subgroup").mean()["r2"].values
            r2_se = df[df["group"] == group].groupby("subgroup").sem()["r2"].values
            axes[i].bar(
                x=np.arange(len(r2)),
                height=r2,
                yerr=r2_se * 2,
                edgecolor="k",
                linewidth=1,
                alpha=0.6,
                color=color,
                width=0.6,
            )
        elif plot_type == "box":
            df_group = df[df.group == group]
            r2 = [
                df_group[df_group["subgroup"] == sg]["r2"].values
                for sg in df_group["subgroup"].unique()
            ]
            props = {"linewidth": 0.65}
            bplot = axes[i].boxplot(
                positions=np.arange(len(r2)),
                x=r2,
                sym="",
                widths=0.23,
                patch_artist=True,
                boxprops=props,
                whiskerprops=props,
                capprops=props,
                medianprops=props,
            )
            for patch in bplot["boxes"]:
                patch.set_facecolor(color)
            for patch in bplot["medians"]:
                patch.set_color("black")

        axes[i].set_xlim(-0.5, len(r2) - 0.5)
        axes[i].set_xticks(np.arange(len(r2)))
        axes[i].set_xlabel(group)
    axes[0].set_ylabel("$R^2 (y, \widehat{y})$", fontsize=12)
    fig.subplots_adjust(wspace=0.1)

    return fig, axes


def plot_group_predint(
    df: pd.DataFrame,
    figsize=(7, 1.8),
    methods: List = None,
    method_colors: Dict = None,
    groups=None,
    pos_offset: float = 0.3,
    widths=0.2,
    legend_bbox_to_anchor=(0.5, 0.96),
    legend_fontsize=10,
    width_ratios=None,
):
    """Plot the prediction interval summary

    Parameters
    ----------
    df : pd.DataFrame
        df contains: 'method', 'group', 'subgroup', 'coverage', 'length'

    pos_offset : float, optional
        position offset
    """
    # plot 2 figures
    if methods is None:
        methods = df["method"].unique()
    n_method = len(methods)
    if method_colors is None:
        palatte = sns.color_palette("Set1", n_method)
        method_colors = {method: color for method, color in zip(methods, palatte)}

    assert len(method_colors) == n_method
    if groups is None:
        groups = df["group"].unique()
    if width_ratios is None:
        width_ratios = (
            np.array([len(df[df["group"] == g]["subgroup"].unique()) for g in groups])
            + 1
        )

    fig_list = []
    axes_list = []
    for val_col in ["coverage", "length"]:
        fig, axes = plt.subplots(
            figsize=figsize,
            ncols=len(width_ratios),
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios},
            dpi=150,
        )

        for i, method in enumerate(methods):
            _group_plot(
                df[df["method"] == method],
                val_col=val_col,
                groups=groups,
                pos_offset=-pos_offset * (len(methods) - 1) / 2 + pos_offset * i,
                axes=axes,
                widths=widths,
                color=method_colors[method],
            )
        legend_elements = [
            Patch(
                facecolor=method_colors[method],
                edgecolor="k",
                label=method,
            )
            for method in methods
        ]
        fig.legend(
            handles=legend_elements,
            loc="center",
            ncol=len(methods),
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            frameon=False,
        )
        if val_col == "coverage":
            axes[0].set_ylabel("Coverage of \nPrediction interval", fontsize=11)
            axes[0].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.0%}".format(y))
            )

        elif val_col == "length":
            axes[0].set_ylabel("Length of \nPrediction interval", fontsize=11)
        fig.subplots_adjust(wspace=0.1)
        fig_list.append(fig)
        axes_list.append(axes)
    return fig_list[0], axes_list[0], fig_list[1], axes_list[1]


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
