__version__ = "0.1.0"

from jax import config
from . import geometry
from . import solvers

from .geometry import Manifold, FinslerMetric
from .geometry.zoo import Euclidean, Riemannian, Randers
from .solvers import AVBDSolver

def enable_x64():
    config.update("jax_enable_x64", True)