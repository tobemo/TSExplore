import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from scipy import stats

seaborn.set_theme()


def distribution(x: pd.Series, *, outliers: bool = False, **kwargs) -> plt.Axes:
    if not isinstance(x, pd.Series):
        raise TypeError(
            f"data should be of type 'pd.Series', is '{type(x)}'."
        )
    
    inliers = x.index
    if not outliers:
        iqr = stats.iqr(x)
        q1, q3 = x.quantile([0.25, 0.75])
        pos_outlier = x.index[ x > q3 + 1.5 * iqr ]
        neg_outlier = x.index[ x < q1 + 1.5 * iqr ]
        outliers = pos_outlier.union(neg_outlier)
        inliers = x.index.difference(outliers)
    
    x = x.loc[inliers]
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

