from typing import Any, Tuple

import torch
import torch.nn as nn

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


class TwoInputModule(nn.Module):
    def forward(self, input1: Any, input2: Any) -> Tuple[Any, Any]:
        raise NotImplementedError("Subclasses must implement the forward method.")


class TwoInputSequential(nn.Sequential, TwoInputModule):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

    def forward(self, input1: Any, input2: Any) -> Any:
        for module in self:
            if isinstance(module, TwoInputModule):
                input1, input2 = module(input1, input2)
            else:
                input1 = module(input1)
        return input1


class Sub_Adder(TwoInputModule):
    def __init__(self, x_dim: int, z_dim: int) -> None:
        super(Sub_Adder, self).__init__()
        self.add_noise = nn.Sequential(
            nn.Linear(z_dim, x_dim),
        )

    def forward(self, input: torch.Tensor, noise: float) -> torch.Tensor:
        # multiplier = self.add_noise.forward(noise)
        multiplier = torch.sigmoid(self.add_noise.forward(noise))
        return input * multiplier
