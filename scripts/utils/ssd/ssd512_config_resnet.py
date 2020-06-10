import numpy as np
import collections
from .ssd_utils import generate_ssd_priors

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
image_size = 512
specs = [
    SSDSpec(64, 8, SSDBoxSizes(20.48, 61.2), [2]),
    SSDSpec(32, 16, SSDBoxSizes(61.2, 133.12), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(133.12, 215.04), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(215.04,  296.96), [2, 3]),
    SSDSpec(4, 128, SSDBoxSizes(296.96, 378.88), [2, 3]),
    SSDSpec(2, 256, SSDBoxSizes(378.88, 460.8), [2]),
    SSDSpec(1, 512, SSDBoxSizes(460.8, 542.72), [2])
]
# specs = [
#     SSDSpec(64, 8, SSDBoxSizes(35.84, 76.8), [2]),
#     SSDSpec(32, 16, SSDBoxSizes(76.8, 153.6), [2, 3]),
#     SSDSpec(16, 32, SSDBoxSizes(153.6, 230.4), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(230.4,  307.2), [2, 3]),
#     SSDSpec(4, 128, SSDBoxSizes(307.2, 384.0), [2, 3]),
#     SSDSpec(2, 256, SSDBoxSizes(384.0, 460.8), [2]),
#     SSDSpec(1, 512, SSDBoxSizes(460.8, 537.6), [2])
# ]


priors = generate_ssd_priors(specs, image_size)