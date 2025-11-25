# Top-level API
from .manifolds import Sphere, Manifold
from .geometry import (
    RandersMetric,
    RandersFactory,
    discrete_randers_energy,
    parallel_transport,
)
from .solvers import ProjectedGradientSolver, AVBDSolver
from .nn import MetricNet, ContextNet
from .embeddings import TokenMap, LearnableTokenMap
from .utils import generate_icosphere
from .losses import holonomy_error_loss
from .models import contrastive_loss, init_encoder_params, grad, apply_encoder
