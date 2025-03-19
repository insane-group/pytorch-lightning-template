<div align="center">
<a href="https://insane.iit.demokritos.gr/">
    <img
      src="https://raw.githubusercontent.com/insane-group/.github/refs/heads/main/images/logo.png"
      alt="LOGO"
    />
  </a>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/1234.56789">
    <img
      src="https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg"
      alt="arXiv"
    />
  </a>
  <a href="https://insane-group.github.io/pytorch-lightning-template/">
    <img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" alt="Mkdocs" />
  </a>
  <a href="https://colab.research.google.com/github/insane-group/pytorch-lightning-template/blob/master/notebooks/template.ipynb">
    <img
      src="https://colab.research.google.com/assets/colab-badge.svg"
      alt="Open In Colab"/>
  </a>
  <a href="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/tests.yml">
    <img
      src="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/tests.yml/badge.svg"
      alt="CI"
    />
  </a>
  <a href="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/pre-commit.yml">
    <img
      src="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/pre-commit.yml/badge.svg"
      alt="pre-commit"
    />
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img
      src="https://img.shields.io/github/license/insane-group/pytorch-lightning-template"
      alt="LICENSE"
    />
  </a>
</p>

## Project Name

> Accompanying code for the paper [**Paper Title**](https://arxiv.org/abs/1234.56789).

Add a brief description of your project here. You can use Markdown syntax for formatting, such as **bold**, _italics_, and [links](https://www.example.com).

Make sure you:

- [ ] Replace the **Project Name** with the name of your project
- [ ] Add a brief description of your project
- [ ] Remove the **:thinking: Why ?** section
- [ ] Rename the project appropriately
- [ ] Change the project details (e.g. name, description, URLs) in the following files:
  - [ ] pyproject.toml
  - [ ] mkdocs.yml
  - [ ] README.md
- [ ] Update CITATION.cff (and the **:bookmark_tabs: Citation** section below)
- [ ] Update the arXiv badge in the README.md with the correct arXiv ID (when available)
- [ ] Update the logo in the README.md

## :thinking: Why ?

When working on a new project, we frequently encountered challenges such as:

1. **Reproducibility**: How can we ensure that our results are reproducible across different environments?
2. **Boilerplate Code**: We often find ourselves writing the same boilerplate code over and over again.

To address these challenges, we have created a template for PyTorch projects that streamlines the setup process and helps you focus on your research.

### Main Technologies

- [**PyTorch Lightning**](https://lightning.ai/docs/pytorch/stable/starter/introduction.html): A lightweight wrapper for PyTorch that streamlines high-performance AI research. It serves as a structured framework for organizing PyTorch code.
- [**Hydra**](https://hydra.cc/): A powerful configuration framework for managing complex applications. It enables dynamic composition of hierarchical configurations, allowing overrides via config files and the command line.

## :rocket: Getting Started

Click [<kbd>Use this template</kbd>](https://github.com/insane-group/pytorch-lightning-template/generate) to create a new repository.

Once your repository is set up [using the template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template), clone it and start working with the following commands (We use the [Rye](https://rye.astral.sh/) Python package manager):

```shell
# Install Rye (https://rye.astral.sh/guide/installation/)
curl -sSf https://rye.astral.sh/get | bash

# Clone the repository & cd into it
git clone https://github.com/insane-group/<YOUR-PROJECT-NAME>
cd <YOUR-PROJECT-NAME>

# Rename the project and make sure you change the project details (e.g. name, description, URLs) in the following files:
# 1. pyproject.toml
# 2. mkdocs.yml
# 3. README.md
mv src/project src/<YOUR-PROJECT-NAME>

# Install dependencies using Rye
rye sync

# Activate the virtual environment
source .venv/bin/activate

# Install the pre-commit hooks
poe hooks

# Run the training/evaluation script
# Override any config parameter from command line
python src/project/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
# train on CPU
python src/project/train.py trainer=cpu
# train on 1 GPU
python src/project/train.py trainer=gpu
# test checkpoint on test dataset
python src/project/test.py checkpoint="/path/to/ckpt/name.ckpt"
```

> Feel free to share any relevant details to help others get started, for example, content similar to the *Setup* and *Quickstart* sections in [Google’s Prompt-to-Prompt](https://github.com/google/prompt-to-prompt?tab=readme-ov-file#setup).

### Performing tasks using `poethepoet`

We are using [poethepoet](https://github.com/nat-n/poethepoet), to perform various development oriented tasks. Formatting, type-checking, as well as a few other operations, can be performed by running

```shell
poe <task>
```

where `<task>` is one of the tasks listed by running:

```shell
poe --help
Poe the Poet - A task runner that works well with poetry.
version 0.28.0

Result: No task specified.

Usage:
  poe [global options] task [task arguments]

Global options:
  -h, --help            Show this help page and exit
  --version             Print the version and exit
  -v, --verbose         Increase command output (repeatable)
  -q, --quiet           Decrease command output (repeatable)
  -d, --dry-run         Print the task contents but don't actually run it
  -C PATH, --directory PATH
                        Specify where to find the pyproject.toml
  -e EXECUTOR, --executor EXECUTOR
                        Override the default task executor
  --ansi                Force enable ANSI output
  --no-ansi             Force disable ANSI output

Configured tasks:
  clean                 Clean up any auxiliary files
  format                Format your codebase
  hooks                 Run all pre-commit hooks
  test                  Run the test suite
  type-check            Run static type checking on your codebase
  lint                  Lint your code for errors
  docs                  Build and serve the documentation
```

> Consider installing `poe` as global dependency to make your life easier using `rye install poethepoet` :stuck_out_tongue:.

## :open_file_folder: Project Structure

The project follows a standard structure for a Python project.

```shell
├── CITATION.cff                                 <- Citation file for referencing the project
├── configs                                      <- Hydra configuration files
│   ├── callbacks                                  <- Configs for training callbacks
│   ├── data                                       <- Configs for loading the dataset
│   ├── hydra                                      <- Hydra-specific settings
│   ├── loggers                                    <- Configs for logging
│   ├── model                                      <- Model-specific config
│   ├── test.yaml                                  <- Test configuration file
│   ├── trainer                                    <- Configs for the training loop
│   └── train.yaml                                 <- Training configuration file
├── data                                         <- Dataset storage directory
├── docs                                         <- Project documentation
│   ├── code                                       <- Source code documentation
│   ├── CODE_OF_CONDUCT.md                         <- Guidelines for community behavior
│   ├── CONTRIBUTING.md                            <- Instructions for contributing to the project
│   ├── images                                     <- Images used in documentation
│   ├── LICENSE.md                                 <- License information
│   ├── index.md                                   <- Main documentation page
│   └── welcome.md                                 <- Welcome page for the project
├── .editorconfig                                <- Editor configuration for consistent formatting
├── .github                                      <- GitHub-specific configurations
│   └── workflows                                  <- CI/CD workflow definitions for GitHub Actions
├── .gitignore                                   <- Files and directories to ignore in Git
├── logs                                         <- Logs generated during training/testing
├── models                                       <- Trained models and related files
├── notebooks                                    <- Jupyter notebooks for experiments and analysis
│   └── template.ipynb                             <- Notebook template for new experiments
├── .pre-commit-config.yaml                      <- Pre-commit hook configurations
├── src                                          <- Source code directory
│   └── project                                    <- Main project codebase
├── tests                                        <- Unit tests for the project
│   ├── __init__.py                                <- Init file for test module
│   └── test_model.py                              <- Tests for model functionality
├── LICENSE                                      <- License information for the project
├── README.md                                    <- Main project README file
├── mkdocs.yml                                   <- Configuration for MkDocs documentation site
├── pyproject.toml                               <- Python project configuration file
├── .python-version                              <- Python version specification
├── requirements-dev.lock                        <- Locked dependencies for development
├── requirements.lock                            <- Locked dependencies for production
└── .vscode                                      <- VS Code workspace settings
    ├── extensions.json                            <- Recommended extensions for VS Code
    ├── launch.json                                <- Debugging configurations
    └── settings.json                              <- VS Code-specific settings
```

## :book: Exploring the Documentation

The documentation is generated from Python docstrings using [`MkDocs`](https://www.mkdocs.org/) and [`mkdocstrings`](https://mkdocstrings.github.io/) for the source code, while the rest is written in standard Markdown. To view it, run `poe docs` in the terminal or visit [`https://insane-group.github.io/pytorch-lightning-template/`](https://insane-group.github.io/pytorch-lightning-template/).

## :bookmark_tabs: Citation

Please use the following citation if you use this project in your work:

```bibtex
@software{Sioros_scikit-learn-template,
  author = {Sioros, Vassilis},
  license = {Apache-2.0},
  title = {{scikit-learn-template}},
  url = {https://github.com/insane-group/scikit-learn-template}
}
```

## :coin: Credits

This template was created by [INSANE Group](https://github.com/orgs/insane-group) and is based on the following projects:

- [**NN-Template by Grok AI**](https://github.com/grok-ai/nn-template)
- [**Lightning Hydra Template by ashleve**](https://github.com/ashleve/lightning-hydra-template)
- [**Pytorch Lightning Template by DavidZhang73**](https://github.com/DavidZhang73/pytorch-lightning-template)
- [**MNIST Classifier by kengz**](https://github.com/kengz/mnist-classifier)
