# HAM Notebooks

This directory contains Jupyter notebooks for demonstrating and experimenting with the HAM library.

## Getting Started

1. Make sure you have activated the virtual environment:
   ```bash
   source ../venv/bin/activate
   ```

2. Start Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. Open `demo.ipynb` to see examples of HAM functionality.

## Available Notebooks

- **demo.ipynb** - Introduction to HAM library with examples of:
  - Sphere manifold operations
  - Exponential and logarithmic maps
  - Finsler geometry with Randers metrics
  - Neural network integration
  - Visualization

## Creating New Notebooks

When creating new notebooks, remember to:
1. Set the Python path to include the `src` directory
2. Force CPU usage if needed: `os.environ['JAX_PLATFORMS'] = 'cpu'`
3. Import HAM modules from `ham.*`

## Tips

- Use `%load_ext autoreload` and `%autoreload 2` to automatically reload modified modules
- The notebooks use JAX, so remember that JAX arrays are immutable
- For visualization, matplotlib is included in the environment
