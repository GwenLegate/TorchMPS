#!/usr/bin/env python3
import time
import torch
from torchmps import PEPS
from torchvision import transforms, datasets

# Miscellaneous initialization
torch.manual_seed(0)
start_time = time.time()

# PEPS parameters
bond_dim = 3

# Training parameters
#input_dim = 32 # cifar
input_dim = 28 # MNIST
num_train = 60000 # 60000 MNIST, 50000 cifar
num_test = 10000
batch_size = 100
num_epochs = 10
learn_rate = 1e-8
l2_reg = 0.0
feature_dim = 2
embedding = 0

# Initialize the MPS module
peps = PEPS(
    input_dim=input_dim,
    output_dim=10,
    bond_dim=bond_dim,
    feature_dim=feature_dim,
)

# Check for GPU, if no GPU, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

peps.to(device)

# Set our loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(peps.parameters(), lr=learn_rate, weight_decay=l2_reg)

# Get the training and test sets
transform = transforms.ToTensor()
#MNIST
train_set = datasets.MNIST("./mnist", download=True, transform=transform)
test_set = datasets.MNIST("./mnist", download=True, transform=transform, train=False)

#fashion MNIST
#train_set = datasets.FashionMNIST("./fashion_mnist", download=True, transform=transform)
#test_set = datasets.FashionMNIST("./fashion_mnist", download=True, transform=transform, train=False)

#cifar10
#train_set = datasets.CIFAR10("./cifar", train=True, download=True, transform=transform)
#test_set = datasets.CIFAR10("./cifar", train=False, download=True, transform=transform)

# Put MNIST data into dataloaders
samplers = {
    "train": torch.utils.data.SubsetRandomSampler(range(num_train)),
    "test": torch.utils.data.SubsetRandomSampler(range(num_test)),
}
loaders = {
    name: torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=samplers[name], drop_last=True
    )
    for (name, dataset) in [("train", train_set), ("test", test_set)]
}
num_batches = {
    name: total_num // batch_size
    for (name, total_num) in [("train", num_train), ("test", num_test)]
}

print(
    f"Training on {num_train} images \n"
    f"(testing on {num_test}) for {num_epochs} epochs"
)
print(f"Maximum PEPS bond dimension = {bond_dim}")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

# Let's start training!
for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels in loaders["train"]:
        try:
            inputs, labels = inputs.view([batch_size, 28 ** 2]).to(device), labels.data.to(device)
        except:
            inputs, labels = inputs.view([batch_size, 32 ** 2, 3]).to(device), labels.data.to(device)

        # Call our MPS to get logit scores and predictions
        scores = peps(inputs, embedding)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / num_batches['train']:.10f}")
    print(f"Average train accuracy: {running_acc / num_batches['train']:.4f}")

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            try:
                inputs, labels = inputs.view([batch_size, 28 ** 2]).to(device), labels.data.to(device)
            except:
                inputs, labels = inputs.view([batch_size, 32 ** 2, 3]).to(device), labels.data.to(device)


            # Call our PEPS to get logit scores and predictions
            scores = peps(inputs, embedding)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")



