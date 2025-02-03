import torch
from src.train.player_postion import train_loop, test_loop
from torch.utils.data import DataLoader
from src.infra.ml.player_position.dataset import GenDataset, PlayerPositionDataset
from src.infra.ml.player_position.model import PlayerPosition

dataset = GenDataset(r".\databases\players.csv")

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"

training_data = PlayerPositionDataset(
    *dataset.get_train_values(), device=device)
test_data = PlayerPositionDataset(*dataset.get_test_values(), device=device)

train_dataloader = DataLoader(training_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)


model = PlayerPosition(input_size=31, output_size=8).to(device)
learning_rate = 3e-4
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate)


epochs = 8
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataloader, optimizer)
    test_loop(model, test_dataloader)

torch.save(model.state_dict(),
           fr'.\player_position.mdl')
