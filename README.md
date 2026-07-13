# DimFlux: Force-Directed Doubly-Additive Line Diagrams
This repository implements a force-directed placement approach for doubly-additive line drawings.

## References
This project is based on the theory and methods described in the following works.

### Primary Reference
```bash
@article{Noehre2026,
  title = {DimFlux: Force-directed additive line diagrams},
  journal = {International Journal of Approximate Reasoning},
  volume = {197},
  pages = {109734},
  year = {2026},
  issn = {0888-613X},
  doi = {https://doi.org/10.1016/j.ijar.2026.109734},
  url = {https://www.sciencedirect.com/science/article/pii/S0888613X2600109X},
  author = {Marcel Nöhre and Dominik Dürrschnabel and Bernhard Ganter and Gerd Stumme},
  keywords = {Formal concept analysis, Force-directed placement, Additive line diagrams, DimDraw, DimFlux},
  abstract = {The visualization of concept lattices is a central problem in the field of Formal Concept Analysis. Force-directed algorithms, as popular in graph drawing, are a promising approach, treating lattice diagrams as physical models, optimizing node positions based on forces derived from the lattice structure. We build on the work of Zschalig, who, however, limited himself to attribute-additive diagrams. We use a more general additivity, in which both the attributes and the objects contribute to the positions of the concept nodes. We replace the planarity enhancer used by Zschalig to obtain a starting diagram for force-directed optimization with the DimDraw algorithm, which generates structured order diagrams on its own. The combination results in DimFlux, an algorithm that leverages the advantages of DimDraw but generates additive diagrams in which readability is increased by maximizing the conflict distance between nodes and non-incident edges.}
}
```

### Supplementary Material
```bash
@misc{Noehre2026Supplementary,
  author = {Nöhre, Marcel and Dürrschnabel, Dominik and Ganter, Bernhard and Stumme, Gerd},
  title = {A Visual Benchmark of DimFlux: Comparison of Line Diagrams for Concept Lattices},
  month = mar,
  year = 2026,
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18936106},
  url = {https://doi.org/10.5281/zenodo.18936106}
}
```

*If you use this code in your work, please cite the above references.*


## Usage

### Install via PyPI
Install the package from [PyPI](https://pypi.org/project/dim-flux/) using pip.

```bash
pip install dim-flux
```

Execute the layout generation using the `dim-flux` command.
```bash
dim-flux
```

### Install from source
Clone the repository and initialize the environment. uv will automatically create a virtual
environment and sync dependencies based on the pyproject.toml. `uv` will automatically create a
virtual environment.

```bash
uv sync
```

Execute the layout generation using uv run to ensure the correct environment context.
```bash
uv run dim-flux
```

The script will prompt you for a path to a `.cxt` file.
To export the node positions add the `--export` flag.
```bash
uv run dim-flux --export
```
The `--export` flag also works with the PyPI install:
```bash
dim-flux --export
```

### Use as a library
DimFlux can also be called directly from Python via `dim_flux.plot`.

```python
from dim_flux import plot

# cxt can be a path to a .cxt file, or a pandas DataFrame representing the incidence matrix
positions = plot('path/to/context.cxt')
```

By default `plot` returns the computed `(x, y)` coordinates in lectic order as a numpy array.
Pass `export=True` to instead write the PDF, GraphML and `.pos` exports to disk.
```python
plot('path/to/context.cxt', export=True)
```

## Configuration
| Category | Parameter | Description |
| :--- | :--- | :--- |
| **Drawings** | `plot_si_graph` | Visualize the Supremum-Infimum graph ($f_{max}$ is highlighted) |
| | `plot_initial_layout` | Visualize the initial layout of the concept lattice |
| | `plot_optimized_layout` | Visualize the final optimized layout of the concept lattice |
| **Annotations** | `si_graph_annotations` | Display annotations in the SI-Graph |
| | `initial_layout_annotations` | Display annotations in the initial drawing |
| | `optimized_layout_annotations` | Display annotations in the final optimized drawing |
| **Forces** | `plot_individual_forces` | Display force vectors for objects and attributes respectively |
| | `plot_combined_forces` | Display combined force vectors |
| | `plot_gradients` | Display gradient vectors at concepts |
| **Dev** | `plot_origin` | Display the origin $(0,0)$ |

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
