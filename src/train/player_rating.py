import torch
import torch.nn as nn

regress_loss = nn.MSELoss()


def train_loop(model, dataloader, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = regress_loss(pred.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model, dataloader, batch_size):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += regress_loss(pred.squeeze(), y)

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss
