import torch
import torch.nn.functional as F
from Mutual_Information import binary_target_helper
import numpy


def train_svm_and_NB(model_cnn, model_reg, model_fusion_NB, model_fusion_svm, train_loader, device, classes):
    model_reg.eval()
    model_cnn.eval()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = binary_target_helper(target, classes).to(device)
        with torch.no_grad():
            cnn_output = model_cnn(data).max(1, keepdim=True)[1]
            reg_output = model_reg(data.reshape(-1, 784)).max(1, keepdim=True)[1].reshape(-1)
        cnn_output = binary_target_helper(cnn_output.reshape(-1), classes).to(device)
        stacked = torch.stack((cnn_output, reg_output), 1).float()
        model_fusion_svm.fit(stacked, target)
        model_fusion_NB.fit(stacked, target)
    return model_fusion_NB, model_fusion_svm

def test_svm_and_NB(model_cnn, model_reg, model_fusion_NB, model_fusion_svm, test_loader, device, classes):
    model_reg.eval()
    model_cnn.eval()
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        target = binary_target_helper(target, classes).to(device)
        with torch.no_grad():
            cnn_output = model_cnn(data).max(1, keepdim=True)[1]
            reg_output = model_reg(data.reshape(-1, 784)).max(1, keepdim=True)[1].reshape(-1)
        cnn_output = binary_target_helper(cnn_output.reshape(-1), classes).to(device)
        stacked = torch.stack((cnn_output, reg_output), 1).float()
        pred_svm = model_fusion_svm.predict(stacked)
        torch_pred_svm = torch.from_numpy(pred_svm).to(device)
        correct = torch_pred_svm.eq(target.view_as(torch_pred_svm)).sum().item()
        print("Correct: {}".format(correct))
        pred_nb = model_fusion_NB.predict(stacked)
        torch_pred_nb = torch.from_numpy(pred_nb).to(device)
        correct = torch_pred_nb.eq(target.view_as(torch_pred_nb)).sum().item()
        print("Correct: {}".format(correct))

    pass


def test_reg(model, device, test_loader, classes):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28)
            target = binary_target_helper(target, classes).to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
    print('\nTest set (LR): Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def test_cnn(model, device, test_loader, classes):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set (CNN): Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def test_fusioNN(model_fusion, model_cnn, model_reg, device, test_loader, classes):
    model_fusion.eval()
    model_cnn.eval()
    model_reg.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = binary_target_helper(target, classes).to(device)
            with torch.no_grad():
                cnn_output = model_cnn(data).max(1, keepdim=True)[1]
                reg_output = model_reg(data.reshape(-1, 784)).max(1, keepdim=True)[1].reshape(-1)
            cnn_output = binary_target_helper(cnn_output.reshape(-1), classes).to(device)
            stacked = torch.stack((cnn_output, reg_output), 1).float()
            output = model_fusion(stacked)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set (fusioNN): Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def train_fusioNN(log_interval, model_fusion, model_cnn, model_reg, device, train_loader, optimizer, epoch, classes):
    model_fusion.train()
    model_reg.eval()
    model_cnn.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = binary_target_helper(target, classes).to(device)
        with torch.no_grad():
            cnn_output = model_cnn(data).max(1, keepdim=True)[1]
            reg_output = model_reg(data.reshape(-1, 784)).max(1, keepdim=True)[1].reshape(-1)
        cnn_output = binary_target_helper(cnn_output.reshape(-1), classes).to(device)
        stacked = torch.stack((cnn_output, reg_output), 1).float()
        fusion_output = model_fusion(stacked)
        loss = F.nll_loss(F.softmax(fusion_output), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch (FNN): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_reg(log_interval, model, device, train_loader, optimizer, epoch, classes):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 28 * 28)
        # change to 1/0
        target = binary_target_helper(target, classes).to(device)
        output = model(data)
        loss = F.nll_loss(F.softmax(output), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print
        if batch_idx % log_interval == 0:
            print('Train Epoch (REG): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_cnn(log_interval, model, device, train_loader, optimizer, epoch, classes):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.softmax(output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch (CNN): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
