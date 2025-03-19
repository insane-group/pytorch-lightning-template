<div align="center">
<a href="https://insane.iit.demokritos.gr/">
    <img
      src="https://insane.iit.demokritos.gr/wp-content/uploads/2024/11/cropped-insane-logo-final-blue-edited.png"
      alt="LOGO"
    />
  </a>
</div>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/">
    <img src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white" alt="Pytorch" />
  </a>
  <a href="https://lightning.ai/docs/pytorch/stable/">
    <img src="https://img.shields.io/badge/-Lightning-ffffff?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDEiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCA0MSA0OCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwLjQ5OTIgMEwwIDEyVjM2TDIwLjUgNDhMNDEgMzZWMTJMMjAuNDk5MiAwWk0xNi45NTAxIDM2LjAwMTZMMTkuMTA4OSAyNi42ODU2TDE0LjI1NDggMjEuODkyTDI0LjA3OTEgMTEuOTk5MkwyMS45MTYzIDIxLjMyOTZMMjYuNzQ0NCAyNi4wOTc2TDE2Ljk1MDEgMzYuMDAxNloiIGZpbGw9IiM3OTJFRTUiLz4KPC9zdmc+Cg==" alt="Pytorch Lightning" />
  </a>
  <a href="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/ci.yml">
    <img
      src="https://github.com/insane-group/pytorch-lightning-template/actions/workflows/ci.yml/badge.svg"
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

______________________________________________________________________

## :thinking: Why ?

When working on a new project, we frequently encountered challenges such as:

1. **Reproducibility**: How can we ensure that our results are reproducible across different environments?
2. **Boilerplate Code**: We often find ourselves writing the same boilerplate code over and over again.

To address these challenges, we have created a template for PyTorch projects that streamlines the setup process and helps you focus on your research.

## :computer: Main Technologies

- [**PyTorch Lightning**](https://github.com/Lightning-AI/pytorch-lightning): A lightweight wrapper for PyTorch that streamlines high-performance AI research. It serves as a structured framework for organizing PyTorch code.
- [**Hydra**](https://github.com/facebookresearch/hydra): A powerful configuration framework for managing complex applications. It enables dynamic composition of hierarchical configurations, allowing overrides via config files and the command line.

## :rocket: Getting Started

Click on [<kbd>Use this template</kbd>](https://github.com/insane-group/pytorch-lightning-template/generate) to initialize new repository. Having created a repository for your project [**using the template**](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template), you can clone it and get started with the following commands:

```shell
# Clone the repository
git clone https://github.com/insane-group/<YOUR-PROJECT-NAME>
cd <YOUR-PROJECT-NAME>

# Install dependencies
rye sync

# Install pre-commit hooks
poe hooks

# Run the trainer
# TODO!: add your own command here
```

## :coin: Credits

This template was created by [INSANE Group](https://github.com/orgs/insane-group) and is based on the following projects:

- [**NN-Template by Grok AI**](https://github.com/grok-ai/nn-template)
- [**Lightning Hydra Template by ashleve**](https://github.com/ashleve/lightning-hydra-template)
- [**Pytorch Lightning Template by DavidZhang73**](https://github.com/DavidZhang73/pytorch-lightning-template)
