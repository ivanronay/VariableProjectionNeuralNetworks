#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: ze3vjn@inf.elte.hu
##  Last modified: 28.05.2024

import torch

'''
Training a pytorch model for a single epoch
'''
def train_single_epoch(dataloader, model, loss_fn, optimizer,device):
    model.train()
    model.train_mode()

    for _, (X, y) in enumerate(dataloader):
        # Predict
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred[:,0], y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(str(model.state_dict()['vp.weight']))


'''
Test a pytorch model using data from the provided dataloader object
'''
def test(dataloader, model, loss_fn, size, device,print_to_std=True):
    num_batches = len(dataloader)
    model.eval()
    model.test_mode()
    test_loss, correct = 0, 0
    TruePos_1, FalsePos_1, FalseNeg_1 = 0, 0, 0
    TruePos_0, FalsePos_0, FalseNeg_0 = 0, 0, 0
    Se_1 = 0
    PosPred_1 = 0
    Se_0 = 0
    PosPred_0 = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pr = torch.squeeze(pred) # for BS=1 not working
            # pr = pred
            # m = torch.nn.Sigmoid()
            test_loss += loss_fn(pr, y).item()
            # pr = m(pr)
            correct += (torch.round(pr) == y).type(torch.float).sum().item()

            TruePos_1 += torch.logical_and(torch.round(pr) == y, y == 1).type(torch.float).sum().item()
            FalsePos_1 += torch.logical_and(torch.round(pr) == 1, y == 0).type(torch.float).sum().item()
            FalseNeg_1 += torch.logical_and(torch.round(pr) == 0, y == 1).type(torch.float).sum().item()

            TruePos_0 += torch.logical_and(torch.round(pr) == y, y == 0).type(torch.float).sum().item()
            FalsePos_0 += torch.logical_and(torch.round(pr) == 0, y == 1).type(torch.float).sum().item()
            FalseNeg_0 += torch.logical_and(torch.round(pr) == 1, y == 0).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if (TruePos_1 + FalseNeg_1) != 0 : Se_1 = TruePos_1 / (TruePos_1 + FalseNeg_1)
    if (TruePos_1 + FalsePos_1) != 0 : PosPred_1 = TruePos_1 / (TruePos_1 + FalsePos_1)
    if (TruePos_0 + FalseNeg_0) != 0 : Se_0 = TruePos_0 / (TruePos_0 + FalseNeg_0)
    if (TruePos_0 + FalsePos_0) != 0 : PosPred_0 = TruePos_0 / (TruePos_0 + FalsePos_0)
    if print_to_std: print(f"Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct, Se_1, PosPred_1, Se_0, PosPred_0


'''
Train pytorch model for a pre-defined number of epochs
'''
def train(model, train_data_loader, test_data_loader, epochs, loss_fcn, optimizer, device, tr_size,te_size,log=True):
    tr_losses = []
    tr_accuracies = []
    te_losses = []
    te_accuracies = []
    se_1 = []
    pos_pred_1 = []
    se_0 = []
    pos_pred_0 = []
    
    for ep in range(epochs):
        if log: print("[EPOCH]: " + str(ep+1) + "/" + str(epochs))
        train_single_epoch(train_data_loader, model, loss_fcn, optimizer,device)
        tr_l, tr_a, _, _, _, _ = test(train_data_loader, model, loss_fcn,tr_size,device, log)
        te_l, te_a, Se_1, PosPred_1, Se_0, PosPred_0 = test(test_data_loader, model, loss_fcn, te_size,device, log)

        tr_losses.append(tr_l)
        tr_accuracies.append(tr_a)

        te_losses.append(te_l)
        te_accuracies.append(te_a)

        se_1.append(Se_1)
        pos_pred_1.append(PosPred_1)
        se_0.append(Se_0)
        pos_pred_0.append(PosPred_0)


    return tr_losses, tr_accuracies, te_losses, te_accuracies, se_1, pos_pred_1, se_0, pos_pred_0
