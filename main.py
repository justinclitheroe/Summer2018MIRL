from time import gmtime, strftime, clock
import torch
import torch.nn as nn
import os
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn import svm, naive_bayes
from CNN import CNN, CNNGood
from train_and_test import train_cnn, train_reg, \
    test_cnn, test_reg, train_fusioNN, test_fusioNN, \
    train_svm_and_NB, test_svm_and_NB
from Mutual_Information import calculate_mutual_info


def train_n_test(epochs, log_interval, model_cnn, model_reg, model_fusion,
                 model_fusion_NB, model_fusion_svm,
                 device, train_loader, test_loader,
                 optimizer_cnn, optimizer_reg, optimizer_fusion,
                 savepath_cnn, savepath_reg, savepath_fusion, save, resume, classes):
    """
    A helper function for testing

    :param epochs:
    :param log_interval:
    :param model_cnn:
    :param model_reg:
    :param model_fusion:
    :param model_fusion_NB:
    :param model_fusion_svm:
    :param device:
    :param train_loader:
    :param test_loader:
    :param optimizer_cnn:
    :param optimizer_reg:
    :param optimizer_fusion:
    :param savepath_cnn:
    :param savepath_reg:
    :param savepath_fusion:
    :param save: Bool: Should the models be saved
    :param resume: Bool: Should the models be trained
    :param classes: List of classes the models are tested on
    :return:
    """
    for epoch in range(1, epochs + 1):
        if resume is False:
            train_cnn(log_interval, model_cnn, device, train_loader, optimizer_cnn, epoch, classes)
            train_reg(log_interval, model_reg, device, train_loader,
                      optimizer_reg, epoch, classes)
        acc_cnn = test_cnn(model_cnn, device, test_loader, classes)
        acc_reg = test_reg(model_reg, device, test_loader, classes)

        # torch.save(model_fusion.cuda().state_dict(), savepath_fusion.format())
    for epoch in range(1, epochs + 1):
        train_fusioNN(log_interval, model_fusion, model_cnn, model_reg, device, train_loader, optimizer_fusion,
                      epoch, classes)
        acc_fusion = test_fusioNN(model_fusion, model_cnn, model_reg, device, test_loader, classes)

    model_fusion_NB, model_fusion_svm = train_svm_and_NB(model_cnn, model_reg, model_fusion_NB, model_fusion_svm,
                                                         train_loader, device, classes)
    test_svm_and_NB(model_cnn, model_reg, model_fusion_NB, model_fusion_svm, test_loader, device, classes)


def print_stats(mi_cnn, mi_reg, mi_joint, mi_redundancy,
                ent_joint, ent_cnn, ent_reg, ent_target,
                acc_cnn, acc_reg, pre_cnn, pre_reg,
                f1_CNN, f1_REG):
    """
    A printing helper function

    see the mutual information function docstring for a description of the parameters
    :return:
    """
    print("I(CNN, target): {:.3f}".format(mi_cnn))
    print("I(REG, target): {:.3f}".format(mi_reg))
    print("I(Joint, target): {:.3f}".format(mi_joint))
    print("I(CNN, REG): {:.3f}".format(mi_redundancy))
    print("H(REG): {:.3f}".format(ent_reg))
    print("H(CNN): {:.3f}".format(ent_cnn))
    print("H(Target): {:.3f}".format(ent_target))
    print("H(joint): {:.3f}".format(ent_joint))
    print("accuracy CNN: {:.3f}".format(acc_cnn))
    print("accuracy REG: {:.3f}".format(acc_reg))
    print("Precision CNN: {:.3f}".format(pre_cnn))
    print("Precision REG: {:.3f}".format(pre_reg))
    print("f1 CNN: {:.3f}".format(f1_CNN))
    print("f1 REG: {:.3f}".format(f1_REG))

    pass


def main():
    """
    A function primarily consisting of set up and hyper parameters

    This function sets up hyperparameters as well as loads or instantiates
    neural networks and other classifiers

    :return:
    """
    single_batch = True
    # Seeding stuff
    seed = 1
    torch.manual_seed(seed)
    # Save handling
    savedirectory = "models/"
    savepath_reg = os.path.join(savedirectory, "trained-regression-ACC{}-T{}")
    savepath_cnn = os.path.join(savedirectory, "trained-CNN-ACC{}-T{}")
    savepath_fusion = os.path.join(savedirectory, "trained-fusion-ACC{}-T{}")
    loadpath_reg = os.path.join(savedirectory, "trained-regression-ACC66-T1731")
    loadpath_cnn = os.path.join(savedirectory, "trained-CNN-ACC66.89-T1730")
    loadpath_fusion = os.path.join(savedirectory, "trained-fusion")
    # hyperparameters for loading and saving
    resume = False
    save = True

    # batch size handling
    batch_size = 100
    if single_batch:
        test_batch_size = 10000
    else:
        test_batch_size = 100
    # hyperparameters
    epochs = 5
    lr = 0.001
    log_interval = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # train/test data set handling
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=test_batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionMnist/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionMnist/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # model instantiation (loading if flag is set)
    # model_cnn = CNN().to(device)
    model_cnn = CNN().to(device)
    # model_fusion = nn.Linear(2, 2).to(device)
    classes = [3, 5, 6, 7, 8]
    model_fusion = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 8),
        nn.Linear(8, 16),
        nn.Linear(16, len(classes) + 1)
    ).to(device)
    model_fusion_NB = naive_bayes.GaussianNB()
    model_fusion_svm = svm.SVC()
    model_reg = nn.Sequential(
        nn.Linear(784,392),
        nn.Linear(392,len(classes) + 1)
        # nn.Linear(392,196),
        # nn.Linear(196,len(classes) + 1)
    ).to(device)
    # model_reg = nn.Linear(784, len(classes) + 1).to(device)
    # Loading
    if resume:
        model_cnn.load_state_dict(torch.load(loadpath_cnn))
        model_cnn.eval()
        model_cnn.to(device)
        model_reg.load_state_dict(torch.load(loadpath_reg))
        model_reg.eval()
        model_reg.to(device)

    # Optimizers
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=lr)
    optimizer_reg = optim.SGD(model_reg.parameters(), lr=lr, momentum=0.9)
    optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=lr/10)
    train_n_test(epochs, log_interval,
                 model_cnn, model_reg, model_fusion,
                 model_fusion_NB, model_fusion_svm,
                 device, train_loader, test_loader,
                 optimizer_cnn, optimizer_reg, optimizer_fusion,
                 savepath_cnn, savepath_reg, savepath_fusion, save, resume, classes)

    # mutual information
    mi_cnn, mi_reg, mi_joint, mi_redundancy, ent_joint, ent_cnn, ent_reg, ent_target, acc_cnn, acc_reg, pre_cnn, pre_reg, f1_CNN, f1_REG = calculate_mutual_info(
        model_cnn, model_reg,
        test_loader, device,
        test_batch_size, classes, single_batch)
    print_stats(mi_cnn, mi_reg, mi_joint, mi_redundancy, ent_joint, ent_cnn, ent_reg, ent_target, acc_cnn, acc_reg,
                pre_cnn, pre_reg, f1_CNN, f1_REG)


if __name__ == '__main__':
    time_start = clock()
    main()
    time_end = clock()
    print(time_end - time_start)
