import torch


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Correcting the shape of the target tensor
            y = y.squeeze()

            pred = model(X).squeeze()  # Also need to fix the output of the model
            test_loss += loss_fn(pred, y).item()

            predicted = (pred > 0.0).float()
            correct += (predicted == y).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")