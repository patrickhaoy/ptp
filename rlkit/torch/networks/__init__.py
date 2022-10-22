"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from rlkit.torch.networks.basic import (
    Sigmoid, Clamp, SigmoidClamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from rlkit.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy, TwoChannelCNN, ConcatTwoChannelCNN
from rlkit.torch.networks.linear_transform import LinearTransform
from rlkit.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)

__all__ = [
    'Sigmoid',
    'Clamp',
    'SigmoidClamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'TwoChannelCNN',
    "ConcatTwoChannelCNN",
    'CNNPolicy',
    'Detach',
    'Flatten',
    'FlattenEach',
    'LinearTransform',
    'MergedCNN',
    'Mlp',
    'Reshape',
    'Split',
]

