The following sections assume that you have already locally [cloned the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).

### Installing Rye

We use the [Rye](https://rye.astral.sh/) Python package manager. [Having installed Rye](https://rye.astral.sh/guide/installation/) you may now create a brand new [virtual environment](https://docs.python.org/3/tutorial/venv.html) and install the project's dependencies by running:

```shell
rye sync
```

### Installing pre-commit hooks

Git hooks are scripts that run automatically to perform tasks like linting and formatting code at different stages of the development process. [pre-commit](https://pre-commit.com/) is a tool designed to manage and share these hooks across projects easily. Having created a virtual environment, install the git hooks by running

```shell
poe hooks
```

### Testing via `pytest`

We are using [pytest](https://github.com/pytest-dev/pytest) to automate testing on multiple environments. The test suite can be run using:

```shell
poe test
```

### Performing tasks using `poethepoet`

We are using [poethepoet](https://github.com/nat-n/poethepoet), to perform various development oriented tasks. Formatting, type-checking, as well as a few other operations, can be performed by running

```shell
poe <task>
```

!!! tip

```
Consider installing `poe` as global dependency to make your life easier using `rye install poethepoet` :stuck_out_tongue:.
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
