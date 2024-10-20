import numpy as np
import pandas as pd
import seaborn
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

seaborn.set_theme(style='white')


def _validate_x(x: pd.Series) -> None:
    if not isinstance(x, pd.Series):
        raise TypeError(
            f"data should be of type 'pd.Series', is '{type(x)}'."
        )


def _drop_outliers(x: pd.Series) -> pd.Series:
    iqr = stats.iqr(x)
    q1, q3 = x.quantile([0.25, 0.75])
    pos_outlier = x.index[ x > q3 + 1.5 * iqr ]
    neg_outlier = x.index[ x < q1 + 1.5 * iqr ]
    outliers = pos_outlier.union(neg_outlier)
    inliers = x.index.difference(outliers)
    
    return x.loc[inliers]


def _have_same_magnitude(*x) -> bool:
    """Returns wether all values passed have the same order of magnitude."""
    magnitude = lambda s: len(str(int(np.median(s[~np.isnan(s)]))))
    magnitudes = [magnitude(_x) for _x in x]
    return len(set(magnitudes)) == 1


def distribution(x: pd.Series, *, outliers: bool = False, **kwargs) -> plt.Axes:
    """Plots distribution of `x`, `log(x)`, `x^2` and `x'`.

    Args:
        x (pd.Series): Input data.
        outliers (bool, optional): Wether to keep outliers. Defaults to False.

    Returns:
        plt.Axes: Axes
    """
    _validate_x(x)

    # drop outliers
    if not outliers:
        x = _drop_outliers(x)
    
    # x transforms
    x_log = np.log(x)
    x_sqr = np.square(x)
    x_diff = np.diff(x)

    # fig
    share_axes = _have_same_magnitude(x, x_log, x_sqr, x_diff)
    height_ratios = [0.36, 0.1, 0.08, 0.36, 0.1] 
    wspace = 0.05
    if not share_axes: 
        height_ratios = [0.33, 0.1, 0.14, 0.33, 0.1]
        wspace = 0.3
    grid = dict(
        height_ratios=height_ratios,
        hspace=0, wspace=wspace,
    )
    fig, axes = plt.subplots(
        nrows=2*2 + 1,
        ncols=2,
        sharex=share_axes,
        sharey='row' if share_axes else False,
        gridspec_kw=grid,
        **kwargs,
    )

    # hide spacer ax (3rd row)
    axes[2,0].set_visible(False)
    axes[2,1].set_visible(False)

    # remove border between hist and boxplot
    for i in [0,3]:
        for j in [0,1]:
            axes[0+i,j].spines['bottom'].set_visible(False)
            axes[1+i,j].spines['top'].set_visible(False)

    # do share x between hist and boxplot
    if not share_axes:
        for a, b in zip(axes[0::2,:].flatten(), axes[1::2,:].flatten()):
            a.sharex(b)
            a.get_xaxis().set_visible(False)
    
    # density plots
    seaborn.kdeplot(x, ax=axes[0,0])
    seaborn.kdeplot(x_log, ax=axes[0,1])
    seaborn.kdeplot(x_sqr, ax=axes[3,0])
    seaborn.kdeplot(x_diff, ax=axes[3,1])
    for ax, title in zip(axes[0::3, :].flatten(), ["x", "log(x)", "x^2", "x'"]):
        ax.set_title(title)
    
    # boxplots
    seaborn.boxplot(x, orient='h', ax=axes[1,0])
    seaborn.boxplot(x_log, orient='h', ax=axes[1,1])
    seaborn.boxplot(x_sqr, orient='h', ax=axes[4,0])
    seaborn.boxplot(x_diff, orient='h', ax=axes[4,1])

    fig.suptitle(
        "Distributions of common transformations"
        f" {'with' if outliers else 'without'} outliers."
    )
    fig.tight_layout()
    return fig


def stationarity(
        x: pd.Series,
        *,
        adf: bool | float = 0.05,
        min_samples: int = 50,
        max_plots: int = 50,
        **kwargs,
    ) -> plt.Axes:
    """Generates stationarity plot. Plots mean and mean +- 1 standard deviation in addition to violin plots over time.

    Args:
        x (pd.Series): Input data.
        adf (bool | float, optional): Wether to run an augmented Dickey-Fuller test. If adf is a float this is used as a threshold for rejecting the null hypothesis. If set to False no adf test is performed which results in a significant speed-up. Defaults to 0.05.
        min_samples (int, optional): Minimum amount of samples required for each violin plot. Influences the number of violin plot that will be generated and takes precedence over `max_plots`. Defaults to 50.
        max_plots (int, optional): The maximum of violin plots generated. Defaults to 50.

    Raises:
        ValueError: Adf falls outside of ]0,1].

    Returns:
        plt.Axes: Axes.
    """
    _validate_x(x)
    if isinstance(adf, float) and (adf <= 0 or adf > 1):
        raise ValueError(
            f"Adf should fall in ]0,1] if float, is '{adf}'."
        )
    
    # min simples takes precedence over max plots
    n = len(x) // max_plots
    n = max(min_samples, n)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        squeeze=False,
        sharex=True,
        sharey='row',
        **kwargs,
    )
    
    y = np.lib.stride_tricks.sliding_window_view(x, window_shape=n)
    y = y[::n]
    y = y.T
    y = pd.DataFrame(y, columns=x.index[(n-1)//2::n])
    seaborn.violinplot(
        data=y,
        fill=False,
        inner='quart',
        color='lightgrey',
        split=True,
        native_scale=True,
        width=0.5,
        ax=axes[0,0],
    )

    mu = x.rolling(n, step=n).mean()
    std = x.rolling(n, step=n).std()
    mu.plot(ax=axes[0,0], color='black')
    (mu + std).plot(ax=axes[0,0], color='grey', linestyle='dashed')
    (mu - std).plot(ax=axes[0,0], color='grey', linestyle='dashed')

    title= 'Stationarity'
    if adf:
        p = adfuller(x)[1]
        thresh = adf if isinstance(adf, float) else 0.05
        is_stat = p < 0.05
        title = f"Data {'is' if is_stat else 'is not'} stationary"
        title += f" (p {p:.2f} < {thresh})."
    
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def report(
        x: pd.Series,
        *,
        outliers: bool = False,
        simple: bool = True,
        adf: bool | float = 0.05,
        min_samples: int = 50,
        max_plots: int = 50,
    ) -> plt.Axes:
    # https://stackoverflow.com/a/70093661
    backend = mpl.get_backend()
    mpl.use('agg')
    dpi = 100

    fig1 = distribution(x=x, outliers=outliers, figsize=(1000/dpi, 1000/dpi))
    if simple:
        adf = False
    fig2 = stationarity(
        x=x,
        adf=adf,
        min_samples=min_samples,
        max_plots=max_plots,
        figsize=(1000/dpi, 1000/dpi)
    )

    c1 = fig1.canvas
    c2 = fig2.canvas

    c1.draw()
    c2.draw()

    a1 = np.array(c1.buffer_rgba())
    a2 = np.array(c2.buffer_rgba())
    a = np.hstack((a1,a2))

    mpl.use(backend)
    fig,ax = plt.subplots(figsize=(2000/dpi, 1000/dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)
    return fig

