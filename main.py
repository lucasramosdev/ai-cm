import sys
import torch
import pickle
import numpy as np
import pandas as pd
from src.infra.ml.player_position.model import PlayerPosition
from src.infra.ml.player_rating.model import PlayerRating
from src.infra.ml.player_position.dataset import labels_to_output
from src.infra.decision_tree.pick_formation.model import labels
from sklearn.preprocessing import StandardScaler

from ollama import chat
from ollama import ChatResponse


def pred_position(device, inputs):
    model = PlayerPosition(input_size=31, output_size=8).to(device)
    model.load_state_dict(torch.load(
        "player_position.mdl", weights_only=True))
    model.eval()
    scaler: StandardScaler = None

    with open("player_position.scl", "rb") as file:
        scaler = pickle.load(file)

    inputs = scaler.transform(inputs.reshape(1, -1))
    inputs = torch.tensor(inputs.astype("float32")).to(device)
    pred = model(inputs)

    del model

    return labels_to_output[pred.argmax(1)]


def pred_rating(device, inputs):
    model = PlayerRating(input_size=31).to(device)
    model.load_state_dict(torch.load(
        "player_rating.mdl", weights_only=True))
    model.eval()

    scaler: StandardScaler = None

    with open("player_rating.scl", "rb") as file:
        scaler = pickle.load(file)

    inputs = scaler.transform(inputs.reshape(1, -1))
    inputs = torch.tensor(inputs.astype("float32")).to(device)
    pred = model(inputs)

    del model

    return pred.item()


def pred_formation(team, formation_type):
    team_input = [team["D"], team["DM"], team["M"],
                  team["AM"], team["F"], formation_type]
    model = None
    with open("pick_formation.mdl", "rb") as file:
        model = pickle.load(file)
    formation = model.predict([team_input])
    return labels[formation[0]]


def add_player_position(team, position):
    if position != "NA":
        team[position] += 1


if len(sys.argv) != 3:
    print("You must provide the CSV path and formation type as argument.")
    sys.exit(1)

csv_path = rf"{sys.argv[1]}"
formation_type = int(sys.argv[2])


df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
allowrd_columns = ["name", "acceleration", "aggression", "agility", "anticipation", "balance", "bravery",
                   "creativity", "crossing", "decisions", "determination", "dribbling", "finishing",
                   "flair", "handling", "heading", "influence", "jumping", "long shots", "marking",
                   "off the ball", "pace", "passing", "positioning", "reflexes", "set pieces",
                   "stamina", "strength", "tackling", "teamwork", "technique", "work rate"]
rest_columns = [
    col for col in allowrd_columns if col in df.columns]
df = df[rest_columns]

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"

if __name__ == "__main__":
    team = {"GK": 0, "D": 0, "DM": 0, "M": 0, "AM": 0, "F": 0}
    ratings = []
    positions = []
    for index, row in df.iterrows():
        inputs = row.drop("name").to_numpy()
        position = pred_position(device, inputs)
        positions.append(position)
        add_player_position(team, position)
        ratings.append(pred_rating(device, inputs))
    df["rating"] = ratings
    df["position"] = positions
    formation = pred_formation(team, formation_type)
    players = df[df["position"] != "NA"][["name", "position", "rating"]]
    message = f"Give me this formation: {formation} with this players: {players.to_string(header=False, index=False)} and 9 substitutes. Choose the players by the rating and position. Answer in Portuguese-BR"
    response: ChatResponse = chat(model="qwen2.5", messages=[
        {
            "role": "system",
            "content": "You are the best player of championship manager 01/02. Send the anwser in Portuguese",
        },
        {
            "role": "user",
            "content": message,
        },
    ])
    print(response.message.content)
