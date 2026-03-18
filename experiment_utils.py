import numpy as np
import pandas as pd
import torch
import yaml
import copy
from typing import Any, Dict, List, Optional
import logging

from data.data_object import DataObject
from model.model_object import ModelObject

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `overrides` into `base`. 
    Values in `overrides` take precedence. Returns a new dict.
    
    Example:
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        overrides = {"b": {"c": 99}, "e": 5}
        result = {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_layer_config(base_config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a layer's base YAML config file and apply any experiment-level overrides.
    
    Args:
        base_config_path: Path to the layer's own config YAML.
        overrides: Dictionary of keys to override from the experiment config.
    
    Returns:
        The merged configuration dictionary.
    """
    base_config = load_yaml(base_config_path)
    if overrides:
        merged = deep_merge(base_config, overrides)
        logger.info(f"Applied {len(overrides)} override(s) to {base_config_path}")
        return merged
    return base_config


def reconstruct_encoding_constraints(
    instance: torch.Tensor, cat_features_indices: List[Any]
) -> torch.Tensor:
    """
    For a given instance, ensure that the one-hot encoded categorical features 
    are valid.
    
    Args:
        instance: The input instance as a torch tensor.
        cat_features_indices: A list of lists indicating the indices of the one-hot encoded categorical features.
    
    Returns:
        The instance with reconstructed encoding constraints.
    """
    x_reconstructed = instance.clone()

    for feature_group in cat_features_indices:
        if isinstance(feature_group, dict):
            features = list(feature_group.get("indices", []))
            encoding = feature_group.get(
                "encoding", "one-hot" if len(features) > 1 else "binary"
            )
            domain = feature_group.get("domain")
        else:
            features = list(feature_group)
            encoding = "one-hot" if len(features) > 1 else "binary"
            domain = None

        if not features:
            continue

        if encoding == "one-hot":
            max_indices_in_group = torch.argmax(x_reconstructed[:, features], dim=1)
            x_reconstructed[:, features] = 0
            row_indices = torch.arange(x_reconstructed.size(0), device=x_reconstructed.device)
            absolute_feature_indices = torch.tensor(
                features, device=x_reconstructed.device
            )[max_indices_in_group]
            x_reconstructed[row_indices, absolute_feature_indices] = 1.0
        elif encoding == "thermometer":
            thermo_values = torch.clamp(x_reconstructed[:, features], 0.0, 1.0)
            levels = torch.round(thermo_values.sum(dim=1)).long()
            levels = torch.clamp(levels, min=1, max=len(features))
            reconstructed = torch.zeros_like(thermo_values)
            for row_idx, level in enumerate(levels.tolist()):
                reconstructed[row_idx, :level] = 1.0
            x_reconstructed[:, features] = reconstructed
        elif encoding == "binary":
            low = 0.0
            high = 1.0
            if domain is not None and len(domain) == 2:
                low = float(domain[0])
                high = float(domain[1])
            midpoint = (low + high) / 2.0
            x_reconstructed[:, features[0]] = torch.where(
                x_reconstructed[:, features[0]] >= midpoint,
                torch.full_like(x_reconstructed[:, features[0]], high),
                torch.full_like(x_reconstructed[:, features[0]], low),
            )
    
    return x_reconstructed


def setup_logging(name: str):
    level = getattr(logging, name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def select_factuals(model: ModelObject, X_test: pd.DataFrame, config) -> pd.DataFrame:
    num_factuals = config.get("num_factuals", 5)
    factual_selection = config.get("factual_selection", "negative_class")

    df = X_test.copy()
        
    if factual_selection == "negative_class":
        df["y"] = model.predict(X_test)
        df = df[df["y"] == 0]
        df = df.drop(columns=["y"]).sample(n=num_factuals, random_state=42)
    elif factual_selection == "all":
        df["y"] = model.predict(X_test)
        df = df[df["y"] == 0]
        df = df.drop(columns=["y"])
    else:
        raise ValueError(f"Unknown factual selection method {factual_selection}")
    
    return df
