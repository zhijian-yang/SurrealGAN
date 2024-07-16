import math
from typing import Any

import torch
import torch.nn.functional as f
from torch.distributions import constraints
from torch.distributions.transforms import Transform

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


class normal_cdf(Transform):
    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, normal_cdf)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.special.ndtr(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.special.ndtri(y)

    def log_abs_det_jacobian(self, x: float, y: float) -> Any:
        return -0.5 * math.log(2 * math.pi) - torch.square(x) / 2.0


def construct_scale_tril(vector: torch.Tensor, ncluster: int) -> torch.Tensor:
    L = torch.zeros((ncluster, ncluster))
    tril_indices = torch.tril_indices(row=ncluster, col=ncluster, offset=0)
    L[tril_indices[0], tril_indices[1]] = vector
    scale_tril = f.normalize(L, dim=1)
    scale_tril[range(ncluster), range(ncluster)] = torch.absolute(
        torch.diag(scale_tril, 0)
    )
    return scale_tril


def construct_corr_matrix(vector: torch.Tensor, ncluster: int) -> torch.Tensor:
    corr_matrix = torch.ones((ncluster, ncluster))
    tril_indices = torch.tril_indices(row=ncluster, col=ncluster, offset=-1)
    triu_indices = torch.triu_indices(row=ncluster, col=ncluster, offset=1)
    corr_matrix[tril_indices[0], tril_indices[1]] = vector
    corr_matrix[triu_indices[0], triu_indices[1]] = vector
    return corr_matrix


def guassian_colula_distribution(
    scale_tril: torch.Tensor, ncluster: int
) -> torch.distributions.TransformedDistribution:
    base_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(ncluster), scale_tril=scale_tril
    )
    return torch.distributions.TransformedDistribution(base_distribution, normal_cdf())
