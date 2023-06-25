## RLSS-23 Practical sessions

This repository contains code and notebooks for the practical sessions of the Reinforcement Learning Summer School 2023.

The suggested way to work is to use Jupyter notebooks as indicated during each practical session.
These are contained in the directory [notebooks](notebooks/). There are two ways to work with notebooks.
The easiest way is to execute them in a Google Colab session.
This allows to run the code cells without the need of installing anything locally.
To do this, open a new [Colab](https://colab.research.google.com/) and choose File -> Open notebook, and insert the GitHub URL of the notebook.
To work locally, instead, follow the instructions in *Local Installation*.

The [rlss_practice](rlss_practice/) package contains environments and auxiliary functions that are used in the notebooks.
This directory also contains some solutions of the exercises we propose. **Do not look into these files if not instructed to do so during the practial session.**


### Local Installation

The source code is provided as a Python package, and it can be simply installed with `pip`.
In Python it is common to work within virtual environments, so that the system installation does not become cluttered and the dependencies stay separate. We also provide instructions for these steps.

The required Python version is 3.9 or 3.10. Verify your Python version by executing the command `python --version`. These versions are currently available in most Linux distributions via the default package manager.
If these are not available, install the required version by following the instructions in [pyenv](https://github.com/pyenv/pyenv).

There are two ways to create the virtual environment and install the package.
The first is with [Poetry](https://python-poetry.org/). Poetry is a tool for developing Python packages. If this is installed in your system, running `poetry install` from the current directory will create a separate virtual environment and install all dependencies. In this case, you may proceed to section *Running*.

The other alternative is to manually create a virtual environment and install the dependencies within.
The commands to create the virtual environment, activate it and then install the package are:

    python -m venv <path>
    source <path>/bin/activate
    pip install .

where `<path>` can be any path of your preference, such as `.venv`

### Running

To execute both the notebooks and parts of this package, it is necessary to enter the virtual environment for each new shell.
Depending on the installation method, the commands are `poetry shell` or `source <path>/bin/activate`.
Now, we can now lauch the notebooks server with `python -m notebook notebooks` from the current directory. Follow the instructions in the browser.