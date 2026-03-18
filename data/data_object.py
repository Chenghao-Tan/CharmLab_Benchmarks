from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml


class DataObject:
    """
    A unified data ingestion and preprocessing pipeline for algorithmic recourse tasks.

    This module reads raw data and a YAML configuration file to dynamically construct
    a processed dataset. It handles feature encoding, scaling, class balancing, and
    data splitting based on the constraints specified in the config.
    """

    def __init__(
        self,
        data_path: str,
        config_path: str = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self._metadata: Dict[str, Dict[str, Any]] = {}

        if config_path is not None:
            with open(config_path, "r") as file:
                self._config = yaml.safe_load(file)
        else:
            self._config = {}

        if config_override is not None:
            self._config = config_override

        self._data_path = data_path
        self.get_preprocessing()

    def get_preprocessing(self) -> None:
        """
        Executes the main preprocessing pipeline based on the YAML configuration.
        """
        self._read_raw_data()
        self._apply_scaling()
        self._apply_encoding()
        self._balance_dataset()
        self._enforce_feature_order()

    def get_processed_data(self) -> pd.DataFrame:
        return self._processed_df

    def set_processed_data(self, new_processed_df: pd.DataFrame) -> None:
        self._processed_df = new_processed_df

    def get_target_column(self) -> str:
        return self._target

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    def get_categorical_features(
        self, expanded: bool = True
    ) -> Union[List[str], List[List[str]]]:
        """
        Returns all discrete input feature groups.

        Despite the historic name, this now includes:
        - binary single-column features
        - one-hot encoded categorical groups
        - thermometer encoded ordinal groups
        """
        groups = self.get_discrete_feature_groups(expanded=expanded)
        if expanded:
            return [group["columns"] for group in groups if group["columns"]]
        return [group["name"] for group in groups]

    def get_discrete_feature_groups(
        self, expanded: bool = True
    ) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for feature, feature_cfg in self._config["features"].items():
            if feature_cfg.get("node_type") != "input":
                continue

            feature_type = feature_cfg.get("type")
            encode = feature_cfg.get("encode")
            if feature_type not in {"binary", "categorical", "ordinal"}:
                continue

            if encode == "one-hot":
                encoding = "one-hot"
            elif encode == "thermometer":
                encoding = "thermometer"
            else:
                encoding = "binary"

            if expanded and encode is not None:
                columns = list(feature_cfg.get("encoded_feature_names") or [])
            else:
                columns = [feature]

            domain = feature_cfg.get("domain")
            if expanded and encode is None and hasattr(self, "_processed_df"):
                if feature in self._processed_df.columns:
                    domain = [
                        float(self._processed_df[feature].min()),
                        float(self._processed_df[feature].max()),
                    ]

            groups.append(
                {
                    "name": feature,
                    "encoding": encoding,
                    "columns": columns,
                    "mutable": bool(feature_cfg.get("mutability", False)),
                    "domain": domain,
                    "ordered_values": list(feature_cfg.get("ordered_values") or []),
                    "category_values": list(feature_cfg.get("category_values") or []),
                }
            )

        return groups

    def get_discrete_feature_groups_with_indices(
        self, columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        feature_order = columns if columns is not None else self.get_feature_names(expanded=True)
        groups_with_indices: List[Dict[str, Any]] = []

        for group in self.get_discrete_feature_groups(expanded=True):
            indices = [
                feature_order.index(column)
                for column in group["columns"]
                if column in feature_order
            ]
            if not indices:
                continue

            groups_with_indices.append({**group, "indices": indices})

        return groups_with_indices

    def get_continuous_features(self) -> List[str]:
        continuous_features = []
        for feature, feature_cfg in self._config["features"].items():
            if (
                feature_cfg.get("type") == "numerical"
                and feature_cfg.get("node_type") == "input"
            ):
                continuous_features.append(feature)
        return continuous_features

    def get_mutable_features(self, mutable: bool = True) -> List[str]:
        mutable_features = []
        for feature, feature_cfg in self._config["features"].items():
            if feature_cfg.get("node_type") != "input":
                continue
            if "mutability" not in feature_cfg:
                raise ValueError(
                    f"Feature '{feature}' is missing the 'mutability' key in the configuration."
                )
            if feature_cfg["mutability"] != mutable:
                continue

            if feature_cfg.get("encode") is not None:
                mutable_features.extend(feature_cfg.get("encoded_feature_names") or [])
            else:
                mutable_features.append(feature)

        return mutable_features

    def _read_raw_data(self):
        self._raw_df = pd.read_csv(self._data_path)
        self._processed_df = self._raw_df.copy()

        columns_to_drop = [
            col for col in self._raw_df.columns if col not in self._config["features"].keys()
        ]
        self._processed_df = self._processed_df.drop(columns=columns_to_drop, errors="ignore")
        self._target = self._config["target_column"]

        for feature in self._config["features"]:
            if feature not in self._raw_df.columns:
                raise ValueError(
                    f"Feature '{feature}' defined in config is not present in the raw dataset."
                )
            self._metadata[feature] = dict(self._config["features"][feature])

    def _apply_encoding(self) -> None:
        for feature, feature_cfg in self._config["features"].items():
            if feature_cfg.get("node_type") != "input":
                continue

            if feature_cfg.get("encode") == "one-hot":
                self._apply_one_hot_encoding(feature)
            elif feature_cfg.get("encode") == "thermometer":
                self._apply_thermometer_encoding(feature)

    def _apply_one_hot_encoding(self, feature_name: str) -> None:
        feature_cfg = self._config["features"][feature_name]
        encoded_feature_names = list(feature_cfg.get("encoded_feature_names") or [])
        category_values = list(feature_cfg.get("category_values") or [])

        if not category_values and encoded_feature_names:
            prefix = f"{feature_name}_cat_"
            category_values = [
                name.replace(prefix, "", 1) if name.startswith(prefix) else name
                for name in encoded_feature_names
            ]

        if not category_values:
            unique_values = self._processed_df[feature_name].astype(str).unique().tolist()
            category_values = unique_values

        if not encoded_feature_names:
            encoded_feature_names = [f"{feature_name}_cat_{i}" for i in range(len(category_values))]

        if len(category_values) != len(encoded_feature_names):
            raise ValueError(
                f"Feature '{feature_name}' has mismatched category_values and encoded_feature_names lengths."
            )

        feature_series = self._processed_df[feature_name].astype(str)
        encoded_columns = {
            encoded_name: (feature_series == str(category_value)).astype(float)
            for category_value, encoded_name in zip(category_values, encoded_feature_names)
        }

        encoded_df = pd.DataFrame(encoded_columns, index=self._processed_df.index)
        self._processed_df = pd.concat(
            [self._processed_df.drop(columns=[feature_name]), encoded_df], axis=1
        )

    def _apply_thermometer_encoding(self, feature_name: str) -> None:
        feature_cfg = self._config["features"][feature_name]
        ordered_values = list(feature_cfg.get("ordered_values") or [])
        encoded_feature_names = list(feature_cfg.get("encoded_feature_names") or [])

        if not ordered_values:
            domain = feature_cfg.get("domain")
            if domain is not None and len(domain) == 2:
                ordered_values = list(range(int(domain[0]), int(domain[1]) + 1))
            else:
                ordered_values = self._processed_df[feature_name].astype(str).unique().tolist()

        if not encoded_feature_names:
            encoded_feature_names = [
                f"{feature_name}_ord_{i}" for i in range(len(ordered_values))
            ]

        if len(ordered_values) != len(encoded_feature_names):
            raise ValueError(
                f"Feature '{feature_name}' has mismatched ordered_values and encoded_feature_names lengths."
            )

        order_lookup = {
            str(value): index for index, value in enumerate(ordered_values)
        }
        feature_values = self._processed_df[feature_name].astype(str)
        thermometer = np.zeros(
            (len(feature_values), len(ordered_values)), dtype=float
        )

        for row_idx, value in enumerate(feature_values):
            if value not in order_lookup:
                raise ValueError(
                    f"Value '{value}' for feature '{feature_name}' is missing from ordered_values."
                )
            thermometer[row_idx, : order_lookup[value] + 1] = 1.0

        encoded_df = pd.DataFrame(
            thermometer, columns=encoded_feature_names, index=self._processed_df.index
        )
        self._processed_df = pd.concat(
            [self._processed_df.drop(columns=[feature_name]), encoded_df], axis=1
        )

    def _apply_scaling(self) -> None:
        strategy = self._config.get("preprocessing_strategy")
        if strategy not in {"normalize", "standardize"}:
            return

        scaler_cls = MinMaxScaler if strategy == "normalize" else StandardScaler

        for feature, feature_cfg in self._config["features"].items():
            if feature_cfg.get("node_type") != "input":
                continue
            if feature_cfg.get("type") not in {"numerical", "binary"}:
                continue
            if feature_cfg.get("encode") is not None:
                continue
            if feature not in self._processed_df.columns:
                continue

            scaler = scaler_cls()
            self._processed_df[feature] = scaler.fit_transform(self._processed_df[[feature]])

    def _balance_dataset(self) -> None:
        if not self._config.get("balance_classes", False):
            return

        target = self._config["target_column"]
        value_counts = self._processed_df[target].value_counts()
        if value_counts.shape[0] != 2:
            raise ValueError("Class balancing currently supports only binary targets.")

        min_count = int(value_counts.min())
        round_to = int(self._config.get("balance_round_to", 250))
        if round_to > 1 and min_count >= round_to:
            sample_size = (min_count // round_to) * round_to
        else:
            sample_size = min_count

        balance_seed = int(self._config.get("balance_seed", 42))
        balanced_parts = [
            self._processed_df[self._processed_df[target] == cls].sample(
                sample_size,
                random_state=balance_seed,
            )
            for cls in sorted(value_counts.index.tolist())
        ]

        self._processed_df = (
            pd.concat(balanced_parts, axis=0)
            .sample(frac=1.0, random_state=balance_seed)
            .reset_index(drop=True)
        )

    def get_train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = self._processed_df.drop(columns=self._config["target_column"])
        y = self._processed_df[self._config["target_column"]]
        split_seed = int(self._config.get("split_seed", 42))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self._config["train_split"],
            random_state=split_seed,
        )
        return X_train, X_test, y_train, y_test

    def _enforce_feature_order(self) -> None:
        post_encoding_feat_order = self._config.get("post_encoding_feat_order")
        if post_encoding_feat_order:
            reordered_columns = [
                column
                for column in post_encoding_feat_order
                if column in self._processed_df.columns
            ]
        else:
            reordered_columns = []
            for feature in self._config.get("feature_order", []):
                feature_cfg = self._config["features"].get(feature, {})
                encoded_feature_names = feature_cfg.get("encoded_feature_names") or []
                if feature_cfg.get("encode") is not None and encoded_feature_names:
                    reordered_columns.extend(
                        [column for column in encoded_feature_names if column in self._processed_df.columns]
                    )
                elif feature in self._processed_df.columns:
                    reordered_columns.append(feature)

        target = self._config["target_column"]
        if target in self._processed_df.columns:
            reordered_columns.append(target)

        self._processed_df = self._processed_df.reindex(columns=reordered_columns)

    def get_feature_names(self, expanded: bool = True) -> List[str]:
        if expanded:
            return [col for col in self._processed_df.columns if col != self._config["target_column"]]

        feature_order = self._config.get("feature_order")
        if feature_order:
            return [
                feature
                for feature in feature_order
                if self._config["features"][feature]["node_type"] == "input"
            ]

        return [
            feature
            for feature, feature_cfg in self._config["features"].items()
            if feature_cfg.get("node_type") == "input"
        ]

    def get_feature_indices(self, feature_name: str) -> List[int]:
        feature_names = self.get_feature_names(expanded=True)
        feature_cfg = self._config["features"].get(feature_name, {})
        if feature_cfg.get("encode") is not None:
            expanded_feature_names = feature_cfg.get("encoded_feature_names") or []
            return [feature_names.index(feature) for feature in expanded_feature_names]
        return [feature_names.index(feature_name)]

    def _filter_and_impute(self) -> None:
        pass

    def inverse_transform(self, x_processed: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Inverse transform is not yet implemented.")
