# Contributing

These guidelines help ensure a structured and effective development process.

## Code of Conduct

By contributing to this project, you agree to uphold the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

As the sole contributor, you should follow a structured workflow that includes Issue tracking and Pull Requests (PRs). Your supervisor may provide feedback on key contributions.

### Issues

Use **Issues** to document problems, propose new features, or note documentation changes. When creating an Issue:

- Clearly describe the problem or feature request.
- If relevant, include error messages, steps to reproduce, or possible solutions.
- Check for existing Issues to avoid duplication.

### Pull Requests

Pull Requests help track changes systematically. When submitting a PR:

- Focus on a single feature or fix.
- Keep changes minimal and well-documented.
- Ensure new functionality includes **unit or integration tests**.
- Update documentation if required.
- Link the PR to an existing Issue if applicable.
- Request a review from your supervisor if the change is significant.

For more details, refer to [this video](https://www.youtube.com/watch?v=nCKdihvneS0). Additionally, feel free to also check [this guide](https://www.atlassian.com/git/tutorials/making-a-pull-request).

### Reviewing Pull Requests

1. **Self-Review**: Before finalizing a PR, review your code for clarity, correctness, and adherence to best practices.
2. **Supervisor Review**: Request feedback from your supervisor.
3. **Address Feedback**: Implement suggested modifications or justify why changes are unnecessary.
4. **Merging PRs**: Once reviewed and approved, merge into the appropriate branch, ensuring project integrity.

### Documenting Your Changes

This project uses [MkDocs](https://www.mkdocs.org/) for documentation, generated from [Python docstrings](https://www.python.org/dev/peps/pep-0257/#id15). When modifying code:

- Update relevant docstrings to reflect changes.
- Run `poe docs` to verify that the documentation builds without errors.

By following these guidelines, you ensure maintainability and clarity in your contributions.
