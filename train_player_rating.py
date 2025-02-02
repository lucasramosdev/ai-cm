import torch
from src.train.player_rating import train_loop, test_loop
from torch.utils.data import DataLoader
from src.infra.ml.player_rating.dataset import GenDataset, PlayerRatingDataset
from src.infra.ml.player_rating.model import PlayerRating
from src.train.utils.early_stop import EarlyStopper


dataset = GenDataset(r".\databases\players.csv")

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"

training_data = PlayerRatingDataset(
    *dataset.get_train_values(), device=device)
test_data = PlayerRatingDataset(*dataset.get_test_values(), device=device)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


model = PlayerRating(input_size=31).to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)


epochs = 100
early_stopper = EarlyStopper(3, 10)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataloader, optimizer, batch_size)
    test_loss = test_loop(model, test_dataloader, batch_size)
    if early_stopper.early_stop(test_loss):
        break
torch.save(model.state_dict(),
           fr'.\player_rating.mdl')
