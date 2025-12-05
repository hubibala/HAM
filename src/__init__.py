__version__ = "0.1.0"

from jax import config
from . import geometry
from . import solvers

from .geometry import Manifold, FinslerMetric
from .geometry.surfaces import Sphere
from .geometry.zoo import Euclidean, Riemannian, Randers, PiecewiseConstantFinsler, DiscreteRanders
from .geometry.mesh import TriangularMesh
from .solvers import AVBDSolver
from .nn import VectorField, PSDMatrixField

def enable_x64():
    config.update("jax_enable_x64", True)