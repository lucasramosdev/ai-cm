import torch
import re
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

labels_to_input = {
    "NA": 0,
    "GK": 1,
    "D": 2,
    "SW": 3,
    "DM": 4,
    "M": 5,
    "AM": 6,
    "F": 7,
    "S": 8,
}

labels_to_output = ["NA", "GK", "D", "SW", "DM", "M", "AM", "F", "S"]


class GenDataset():
    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = None):
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        allowrd_columns = ["ability", "acceleration", "aggression", "agility", "anticipation", "balance", "bravery",
                           "creativity", "crossing", "decisions", "determination", "dribbling", "finishing",
                           "flair", "handling", "heading", "influence", "jumping", "long shots", "marking",
                           "off the ball", "pace", "passing", "positioning", "reflexes", "set pieces",
                           "stamina", "strength", "tackling", "teamwork", "technique", "work rate", "position"]
        rest_columns = [
            col for col in allowrd_columns if col in df.columns]
        df = df[rest_columns]
        df.dropna(inplace=True)
        df.drop_duplicates(subset=rest_columns)
        df['labels'] = df["position"].apply(
            lambda x: re.split(r"[ /]", x, 1)[0])
        df["labels"] = df.apply(
            lambda row: row["labels"] if row["ability"] > 70 else "NA", axis=1)
        features = df.drop(
            columns=["ability", "labels", "position"]).copy(True).to_numpy()
        scaler = preprocessing.StandardScaler().fit(features)
        with open('player_position.scl', 'wb') as file:
            pickle.dump(scaler, file)
        features = scaler.transform(features)
        df["labels"] = df["labels"].apply(
            lambda x: labels_to_input[x])
        labels = df[["labels"]].to_numpy().squeeze()
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            features, labels, test_size=test_size, random_state=random_state)

    def get_train_values(self):
        return self.train_features, self.train_labels

    def get_test_values(self):
        return self.test_features, self.test_labels


class PlayerPositionDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature.astype("float32")).to(self.device), torch.tensor(label.astype("float32")).to(dtype=torch.long, device=self.device)
