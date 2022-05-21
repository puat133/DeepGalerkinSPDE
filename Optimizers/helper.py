from enum import Enum


class OptimizerType(Enum):
    HESSIAN_FREE = 0
    ADA_HESSIAN = 1
    ADAM = 2
    RMSPROP = 3
    RMSPROP_MOMENTUM = 4
    LEVENBERG_MARQUART = 5
    # NESTEROV = 6
