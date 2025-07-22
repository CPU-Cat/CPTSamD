# CPTSamD

Charged Particle Trajectory Simulation and analysis in a magnetic Diverter field

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16331909.svg)](https://doi.org/10.5281/zenodo.16331909)

## Overview

CPTSamD is a Python package for simulating and analyzing the trajectories of charged particles (such as electrons, protons, and antiprotons) in the presence of complex magnetic field geometries, such as those produced by magnetic diverters. The code is designed for scientific research, including detector design, hit rate analysis, and visualization of particle motion.

## Features

- Simulation of charged particle trajectories using realistic magnetic field models (via [magpylib](https://magpylib.readthedocs.io/)).
- Support for multiple particle types and energies.
- Batch processing of simulation data and automated data gathering.
- Hit rate and detector efficiency analysis.
- Visualization tools for 2D and 3D trajectory plots, hit distributions, and detector geometry.
- Modular code for easy extension and adaptation to new geometries or analysis tasks.
- Output of results in both graphical (PNG, EPS) and text (TXT, pickle) formats.

## Project Structure

```
CPTSamD/
    __init__.py
    <simulation and analysis scripts>
    tests/
        __init__.py.txt
pyproject.toml
poetry.lock
README.md
LICENSE
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/CPU-Cat/CPTSamD.git
   cd CPTSamD
   ```

2. **Install dependencies using Poetry:**
   ```sh
   poetry install
   ```

3. **(Optional) Export requirements.txt:**
   ```sh
   poetry export -f requirements.txt --output requirements.txt --without-hashes
   ```

## Usage

Simulation and analysis scripts are located in the `CPTSamD/` directory. Example scripts include:

- `fixedshadeincl_singlepfile_hitrate_analysis_False90_1kelectron_straightdown_iniposcheck.py`: Analyzes hit rates and initial positions for single-energy electron runs.
- `pickle_gather_single_energy_Electrons_LOW.py`: Gathers and consolidates simulation data from multiple runs into a single pickle file.
- `multiedit_RUNCODE_Electrons_25_posset_straightdown_LOWERenergy.py`: Main simulation script for generating electron trajectories with various initial conditions.

To run a script, use:
```sh
poetry run python CPTSamD/<script_name>.py
```

## Dependencies

- Python >= 3.10
- magpylib >= 5.1.1
- numpy
- matplotlib
- scipy

(Install all dependencies with `poetry install`.)

## Documentation

Each script contains detailed docstrings describing its purpose, usage, and changelog. Please refer to the top of each file for more information.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or documentation improvements.

## License

This project is licensed under the GNU GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Contact

Author: Catriana K. Paw U  
Email: ckpawu@bu.edu

---

For more information, see the code and docstrings in the `CPTSamD/` directory.


# CPTSamD Full Documentation can be found here: [full_docs](docs/docs.md)
