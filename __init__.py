import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

seaborn.set_theme()


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


def distribution(x: pd.Series, *, outliers: bool = False, **kwargs) -> plt.Axes:
    """Plots distribution of `x`, `log(x)`, `x^2` and `x'`.

    Args:
        x (pd.Series): Input data.
        outliers (bool, optional): Wether to keep outliers. Defaults to False.

    Returns:
        plt.Axes: Axes
    """
    _validate_x(x)

    if not outliers:
        x = _drop_outliers(x)
    
    x_log = np.log(x)
    x_sqr = np.square(x)
    x_diff = np.diff(x)

    grid = dict(height_ratios=[0.4, 0.1, 0.4, 0.1])
    fig, axes = plt.subplots(
        nrows=2*2,
        ncols=2,
        sharex=True,
        sharey='row',
        gridspec_kw=grid,
        **kwargs,
    )
    seaborn.kdeplot(x, ax=axes[0,0])
    seaborn.kdeplot(x_log, ax=axes[0,1])
    seaborn.kdeplot(x_sqr, ax=axes[2,0])
    seaborn.kdeplot(x_diff, ax=axes[2,1])
    for ax, title in zip(axes[0::2, :].flatten(), ["x", "log(x)", "x^2", "x'"]):
        ax.set_title(title)
    
    seaborn.boxplot(x, orient='h', ax=axes[1,0])
    seaborn.boxplot(x_log, orient='h', ax=axes[1,1])
    seaborn.boxplot(x_sqr, orient='h', ax=axes[3,0])
    seaborn.boxplot(x_diff, orient='h', ax=axes[3,1])

    fig.suptitle(
        "Distributions of common transformations"
        f" {'with' if outliers else 'without'} outliers."
    )
    fig.tight_layout()
    return axes


def stationarity(
        x: pd.Series,
        *,
        outliers: bool = False,
        adf: bool | float = 0.05,
        min_samples: int = 50,
        max_plots: int = 50,
        **kwargs,
    ) -> plt.Axes:
    """Generates stationarity plot. Plots mean and mean +- 1 standard deviation in addition to violin plots over time.

    Args:
        x (pd.Series): Input data.
        outliers (bool, optional): Wether to keep outliers. Defaults to False.
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

    if not outliers:
        x = _drop_outliers(x)
    
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
    return axes

