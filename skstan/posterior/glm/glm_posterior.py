from dataclasses import dataclass

import numpy as np

from skstan.posterior import Posterior


class GlmPosterior(Posterior):
    pass


@dataclass(frozen=True)
class LinearRegressionPosterior(GlmPosterior):
    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        assert self.alpha.ndim == 2
        assert self.beta == 1
        assert self.sigma == 1
