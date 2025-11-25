# HAM: Holonomic Association Model

A modern Python library for differential geometry and deep learning on manifolds.

## Features

- **Manifold Implementations**: Sphere, hyperbolic spaces, and custom manifolds
- **Finsler Geometry**: Support for direction-dependent metrics
- **Deep Learning Integration**: Compatible with PyTorch and JAX
- **Type Hints**: Fully typed for better IDE support and code quality

## Installation

### Basic Installation

```bash
pip install ham
```

### Development Installation

```bash
git clone https://github.com/yourusername/ham.git
cd ham
pip install -e ".[dev]"
```

### With Deep Learning Frameworks

```bash
# For PyTorch support
pip install "ham[torch]"

# For JAX support
pip install "ham[jax]"

# Install everything
pip install "ham[all]"
```

## Quick Start

```python
import numpy as np
from ham.manifolds import Sphere

# Create a 2-sphere (surface of a ball in 3D)
sphere = Sphere(dim=2)

# Points on the sphere
x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])

# Compute the logarithmic map (tangent vector from x to y)
v = sphere.log(x, y)

# Compute the exponential map (move along the tangent vector)
z = sphere.exp(x, v)

print(f"Distance approximation: {np.linalg.norm(v)}")
```

## Project Structure

```
src/
└── ham/
    ├── __init__.py
    ├── manifolds/          # Manifold implementations
    │   ├── base.py         # Abstract base class
    │   └── sphere.py       # Sphere manifold
    └── geometry/           # Geometric structures
        └── finsler.py      # Finsler metrics
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this library in your research, please cite:

```bibtex
@software{ham2025,
  title={HAM: Holonomic Association Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ham}
}
```

## Roadmap

- [ ] Hyperbolic space implementations (Poincaré ball, hyperboloid)
- [ ] Product manifolds
- [ ] Parallel transport
- [ ] Geodesic computations
- [ ] PyTorch and JAX backend integration
- [ ] Optimization on manifolds (Riemannian gradient descent)
- [ ] Neural network layers for manifold-valued data
