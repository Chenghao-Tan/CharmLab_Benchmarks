from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data.data_object import DataObject


def load_adult_data_new(data_dir: Path) -> pd.DataFrame:
    attrs = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
    ]
    sensitive_attrs = {"sex"}
    attrs_to_ignore = {"sex", "race", "fnlwgt"}
    data_files = ["adult.data", "adult.test"]

    y = []
    x_control = {}
    attrs_to_vals = {}
    for attr in attrs:
        if attr in sensitive_attrs:
            x_control[attr] = []
        elif attr not in attrs_to_ignore:
            attrs_to_vals[attr] = []

    for file_name in data_files:
        full_file_name = data_dir / file_name
        with full_file_name.open("r") as handle:
            for line in handle:
                line = line.strip()
                if line == "":
                    continue

                parts = line.split(", ")
                if len(parts) != 15 or "?" in parts:
                    continue

                class_label = parts[-1]
                if class_label in {"<=50K.", "<=50K"}:
                    class_label = 0
                elif class_label in {">50K.", ">50K"}:
                    class_label = 1
                else:
                    raise ValueError(f"Invalid class label value: {class_label}")

                y.append(class_label)

                for index in range(len(parts) - 1):
                    attr_name = attrs[index]
                    attr_val = parts[index]

                    if attr_name == "native_country":
                        if attr_val != "United-States":
                            attr_val = "Non-United-Stated"
                    elif attr_name == "education":
                        if attr_val in {"Preschool", "1st-4th", "5th-6th", "7th-8th"}:
                            attr_val = "prim-middle-school"
                        elif attr_val in {"9th", "10th", "11th", "12th"}:
                            attr_val = "high-school"

                    if attr_name in sensitive_attrs:
                        x_control[attr_name].append(attr_val)
                    elif attr_name not in attrs_to_ignore:
                        attrs_to_vals[attr_name].append(attr_val)

    frame_dict = dict(attrs_to_vals)
    frame_dict["sex"] = x_control["sex"]
    frame_dict["label"] = y
    df = pd.DataFrame.from_dict(frame_dict)

    processed_df = pd.DataFrame()
    processed_df["Label"] = df["label"].astype(int)
    processed_df["Sex"] = df["sex"].map({"Male": 1, "Female": 2}).astype(int)
    processed_df["Age"] = pd.to_numeric(df["age"], errors="raise").astype(int)
    processed_df["NativeCountry"] = (
        df["native_country"]
        .map({"United-States": 1, "Non-United-Stated": 2})
        .astype(int)
    )
    processed_df["WorkClass"] = df["workclass"].astype(str)
    processed_df["EducationNumber"] = (
        pd.to_numeric(df["education_num"], errors="raise").astype(int)
    )
    processed_df["EducationLevel"] = df["education"].astype(str)
    processed_df["MaritalStatus"] = df["marital_status"].astype(str)
    processed_df["Occupation"] = df["occupation"].astype(str)
    processed_df["Relationship"] = df["relationship"].astype(str)
    processed_df["CapitalGain"] = (
        pd.to_numeric(df["capital_gain"], errors="raise").astype(float)
    )
    processed_df["CapitalLoss"] = (
        pd.to_numeric(df["capital_loss"], errors="raise").astype(float)
    )
    processed_df["HoursPerWeek"] = (
        pd.to_numeric(df["hours_per_week"], errors="raise").astype(int)
    )

    return processed_df


class AdultData(DataObject):
    def __init__(
        self,
        data_path: str,
        config_path: str = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(data_path, config_path, config_override)

    def _read_raw_data(self):
        data_dir = Path(self._data_path).resolve().parent
        self._raw_df = load_adult_data_new(data_dir)
        self._processed_df = self._raw_df.copy()
        self._target = self._config["target_column"]

        columns_to_drop = [
            col
            for col in self._raw_df.columns
            if col not in self._config["features"].keys()
        ]
        self._processed_df = self._processed_df.drop(columns=columns_to_drop, errors="ignore")

        for feature in self._config["features"]:
            if feature not in self._raw_df.columns:
                raise ValueError(
                    f"Feature '{feature}' defined in config is not present in the raw dataset."
                )
            self._metadata[feature] = dict(self._config["features"][feature])
