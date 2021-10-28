# All file imports
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os


def parse_args():
    '''process arguments from the command line'''
    parser = argparse.ArgumentParser(
        description='Build and Train your Neural Network')
    parser.add_argument('data_dir', type=str,
                        help='directory of the training data (required)')
    parser.add_argument('--save_dir', type=str,
                        help='directory where to save your neural network. By default it will save in current directory')
    parser.add_argument(
        '--arch', type=str, help='models to pretrain from (vgg13, vgg19, densenet)')
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate of training')
    parser.add_argument('--hidden_units', type=int,
                        help='number of hidden units of the network')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU for training when available')
    # parse arguments
    args = parser.parse_args()
    return args


def get_data(data_dir):
    '''acquiring the dataset'''
    # data_dir = 'data/flowers/'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    # Composing transforms
    data_t = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    test_t = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    train_set = datasets.ImageFolder(train_dir, transform=data_t)
    test_set = datasets.ImageFolder(test_dir, transform=test_t)
    valid_set = datasets.ImageFolder(valid_dir, transform=test_t)

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=64)

    loaders = {
        'train': trainloader,
        'test': testloader,
        'valid': validloader
    }

    return loaders


def build_model(arch=None, hidden_units=None):
    '''build the deep neural network model'''

    if arch is None:
        arch = 'vgg'
    hidden_units = 4096 if hidden_units is None else int(hidden_units)

    # choosing architecture
    if arch == 'vgg':
        model = models.vgg13_bn(pretrained=True)
        input_units = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units = 9216
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_units = 1024

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # creating new classifier
    classifier = nn.Sequential(
        nn.Linear(input_units, int(hidden_units)),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model


def train_model(
        model,
        trainloader,
        validloader,
        learning_rate,
        epochs,
        device):
    '''train the deep neural network model on the data'''

    print_every = 10
    learning_rate = .001 if learning_rate is None else float(learning_rate)
    epochs = 10 if epochs is None else int(epochs)
    if device is None:
        device = torch.device('cpu')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=learning_rate
    )

    steps = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()

        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Val loss: {valid_loss/len(validloader):.3f}.. "
                    f"Val accuracy: {accuracy/len(validloader):.3f}"
                )

                running_loss = 0
                model.train()

    return model


def test_model(model, testloader, device=None):
    '''test the deep neural network model accuracy on new data'''

    if device is None:
        device = torch.device('cpu')
    # Do validation on the test set
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            logps = model.forward(inputs)
            # Calculate accuracy
            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    res = accuracy/len(testloader)
    print(f"Model Accuracy: {res:.3f}")
    return res


def save_model(model, save_dir=None):
    '''method to save a the trained deep neural network model'''
    #  Save the checkpoint
    # model.class_to_idx = image_datasets.class_to_idx
    model_path = None
    if save_dir is None:
        model_path = 'trained_model.pth'
    else:
        model_path = f'{save_dir}/trained_model.pth'

    checkpoint = {
        # 'arch': 'vgg13_bn',
        # 'input_size': 25088,
        # 'output_size': 102,
        # 'hidden_units': 4096,
        'model': model.to(torch.device('cpu')),
        'features': model.features,
        'classifier': model.classifier,
        # 'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        # 'idx_to_class': {val: key for key, val in image_datasets.class_to_idx.items()}
    }

    torch.save(checkpoint, model_path)
    return model_path


def main():
    '''Building and training of a deep neural network model with all the option 
       provided. Trained model will be save in the provided save directory'''

    print('Validating arguments...')
    args = parse_args()

    if(not os.path.isdir(args.data_dir)):
        raise Exception(
            '[ERROR] data_dir option: Data directory does not exist')

    if(args.save_dir is not None and not os.path.isdir(args.save_dir)):
        raise Exception(
            '[ERROR] save_dir option: Save directory provided does not exist')

    data_dir = os.listdir(args.data_dir)
    if (not set(data_dir).issubset({'test', 'train', 'valid'})):
        raise Exception('[ERROR] Missing test, train or valid sub-directories')

    if args.arch not in ('vgg', 'alexnet', 'densenet', None):
        raise Exception(
            '[ERROR] --arch option: Unsupported model. Supported: vgg, alexnet, and densenet.')

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.gpu:
        device = torch.device("cpu")
    else:
        raise Exception("--gpu option: [Error] No GPU found")

    # running the main training
    print('All arguments valid!\nProcessing the datasets...')
    loaders = get_data(data_dir)
    print('Datasets processed!\nBuilding the deep neural network model...')
    model = build_model(args.arch, args.hidden_units)
    print('Model built!\ntraining the model...')
    trained_model = train_model(
        model,
        loaders['train'],
        loaders['valid'],
        args.learning_rate,
        args.epochs,
        device
    )
    print('Model trained!\nTesting model accuracy...')
    test_model(
        trained_model,
        loaders['test'],
        device
    )
    print('Saving model...')
    model_path = save_model(trained_model, args.save_dir)
    print(f'All training complete!\nTrained model saved as {model_path}.')


if __name__ == '__main__':
    main()
