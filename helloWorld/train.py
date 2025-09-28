

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train() # Переводим модель в режим обучения
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Выполняем предсказание и вычисляем ошибку (loss)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Выполняем обратное распространение ошибки (backpropagation)
        # для обновления весов
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")