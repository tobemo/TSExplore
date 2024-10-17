import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from __init__ import distribution, report, stationarity



if __name__ == '__main__':
    n = 10_000
    data=norm.rvs(size=n)
    # data=norm.rvs(size=n) + np.arange(n) # non-stationary
    data = pd.Series(
        data=data,
        index=pd.date_range(
            start='2000',
            periods=n,
            freq='h',
        )
    )

    report(data, outliers=True, simple=False)
    plt.show()
