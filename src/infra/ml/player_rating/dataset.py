import torch
import re
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle


class GenDataset():
    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = None):
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        allowrd_columns = ["ability", "acceleration", "aggression", "agility", "anticipation", "balance", "bravery",
                           "creativity", "crossing", "decisions", "determination", "dribbling", "finishing",
                           "flair", "handling", "heading", "influence", "jumping", "long shots", "marking",
                           "off the ball", "pace", "passing", "positioning", "reflexes", "set pieces",
                           "stamina", "strength", "tackling", "teamwork", "technique", "work rate"]
        rest_columns = [
            col for col in allowrd_columns if col in df.columns]
        df = df[rest_columns]
        df.dropna(inplace=True)
        df.drop_duplicates(subset=rest_columns)
        features = df.drop(
            columns=["ability"]).copy(True).to_numpy()
        scaler = preprocessing.StandardScaler().fit(features)
        with open('player_rating.scl', 'wb') as file:
            pickle.dump(scaler, file)
        features = scaler.transform(features)
        labels = df["ability"].to_numpy().squeeze()
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            features, labels, test_size=test_size, random_state=random_state)

    def get_train_values(self):
        return self.train_features, self.train_labels

    def get_test_values(self):
        return self.test_features, self.test_labels


class PlayerRatingDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature.astype("float32")).to(self.device), torch.tensor(label.astype("float32")).to(device=self.device)
