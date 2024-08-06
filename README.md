# Water-Access
---


<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6866654.svg)](https://doi.org/10.5281/zenodo.6866654) -->
![Testing](https://github.com/allfed/water-access/actions/workflows/testing.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---
The Water Access model is a tool that simulates global access to freshwater sources by walking or cycling in the event of global catastrophic industry loss (GCIL). The output of the model be used to identify areas at various resolutions (currently country-level and district-level) which will be at highest risk of water insecurity in a catastrophe. This model can be useful for researchers, environmentalists, and disaster response teams who are interested in understanding the potential impact of GCIL on water access. The model is implemented in Python and its outputs have the potential to be integrated into other analysis and visualization tools.

## Installation
To install the Water Access package, we recommend setting up a virtual environment. This will ensure that the package and its dependencies are isolated from other projects on your machine, which can prevent conflicts and make it easier to manage your dependencies. Here are the steps to follow:

* Create a virtual environment using either conda by running the command `conda env create -f environment.yml`. This will create an environment called "water-access". A virtual environment is like a separate Python environment, which you can think of as a separate "room" for your project to live in, it's own space which is isolated from the rest of the system, and it will have it's own set of packages and dependencies, that way you can work on different projects with different versions of packages without interfering with each other.

* Activate the environment by running `conda activate water-access`. This command will make the virtual environment you just created the active one, so that when you run any python command or install any package, it will do it within the environment.

* Install the package by running `pip install -e .` in the main folder of the repository. This command will install the package you are currently in as a editable package, so that when you make changes to the package, you don't have to reinstall it again.

* The code is split between python files and Jupytper notebooks. If you want to run the Jupyter notebooks, you'll need to create a kernel for the environment. First, install the necessary tools by running `conda install -c anaconda ipykernel`. This command will install the necessary tools to create a kernel for the Jupyter notebook. A kernel is a component of Jupyter notebook that allows you to run your code. It communicates with the notebook web application and the notebook document format to execute code and display the results.

* Then, create the kernel by running `python -m ipykernel install --user --name=water-access`. This command will create a kernel with the name you specified "water-access" , which you can use to run the example notebook or play around with the model yourself.

* Alternatively, you may wish to run the notebook in an IDE such as Visual Studio Code (instructions [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks))

If you are using the kernel and it fails due an import error for the model package, you might have to rerun: `pip install -e .`.

If you encounter any issues, feel free to open an issue in the repository.

## How this model works in general

In general, this notebook uses the Lankford walking model and Martin cycling model to simulate water access. The mobility and sensitivity notebooks explore the models, the global files apply the models at a 5 arcminute resoltion worldwise, and the Monte Carlo files run simulations of the global model while varying key parameters.

## Getting the data

The data is stored using [Git Large File Storage (LFS)](https://git-lfs.com/).

### Pickle Format

The model simulation results are stored in the pickle format to ensure a quick read time, as the overall results are several gigabytes large. Learn more about pickle [here](https://www.youtube.com/watch?v=Pl4Hp8qwwes).

## Structure

### Processing scripts

To be included.

### The walking and cycling models

The implementation and expxloration of the Lankford (walking) and Martin (cycling) models are located in these files:

* `src/mobility_module.py`: Implementation and helper files for the Lankford and Martin mobility models.

* `scripts/mobility_notebook.ipynb`: Explores running different models with a single set of parameters.

* `scripts/sensitivity_notebook.ipynb`: Sensitivity analysis to formally analyze the impact of changin key input parameters.


#### Global model

Applies the Lankford and Martin models at a global scale:


* `src/gis_global_module.py`: Contains all key functions to apply the walking and cycling models at the global scale, using the models as defined in src/mobility_module.py

* `scripts/gis_global_analysis.ipynb`: Applies the models globally step-by-step. Not used to generate final results, but useful to understand each part of the process.

* `src/gis_monte_carlo.py`: Contains all key functions to run multiple simulations of the functions defined in src/gis_global_module.py

* `scripts/run_monte_carlo.ipynb`: Performs Monte Carlo simulations of the global model runs using functions defined in src/gis_monte_carlo.py

#### Results & plotting

* `scripts/key_results.ipynb`: Generates key results (data and plots) used in publication.