from typing import Any
import hydra


def instantiate_objects(cfg: Any) -> list[Any]:
    """
    Given a configuration with multiple objects, instantiate each one and return the list of objects.

    Args:
        cfg (Any): The configuration to instantiate objects from.

    Returns:
        List[Any]: The list of instantiated objects.
    """
    objects = []
    for _, params in cfg.items():
        objects.append(hydra.utils.instantiate(params))

    return objects
