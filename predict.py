# All file imports
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image


def parse_args():
    '''parse the arguments for the predict application'''
    parser = argparse.ArgumentParser(
        description='Classify an image using your neural network')
    parser.add_argument('image_path', type=str,
                        help='path to the input image to classifier (required)')
    parser.add_argument('checkpoint', type=str,
                        help='path to the model checkpoint (required)')
    parser.add_argument('--top_k', type=int,
                        help='number of top classes to show (default 5)')
    parser.add_argument('--category_names', type=str,
                        help='json file for category names')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU for prediction when available')
    # parse arguments
    args = parser.parse_args()
    return args


def preprocess(image_path):
    '''preprocess the image before feeding it to the model'''
    #  Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)

    # Creating the preprocessor transform
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return preprocessing(img)


def load_model(model_path):
    ''''load model checkpoint from the model_path'''
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint, model


def predict(image_path, model_path, topk=None, device=None):
    '''return the predictions from the model'''

    if topk is None:
        topk = 5
    if device is None:
        device = torch.device('cpu')

    with torch.no_grad():
        # get image
        img = preprocess(image_path)
        img.unsqueeze_(0)
        img.float()
        img = img.to(device)

        # get model
        _, model = load_model(model_path)
        model.to(device)

        outputs = model(img)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()

        return zip(probs, classes)


def print_pred(predictions, cat_file_path=None):
    '''print predictions on the output'''
    k = 0
    for prob, cl in predictions:
        k += 1
        prob = f'{round(prob, 4) * 100.}%'
        if cat_file_path:
            with open(cat_file_path, 'r') as cf:
                cat_to_name = json.load(cf)
                cl = cat_to_name[str(cl)]
        else:
            cl = f'class {cl}'

        print(f'{k}. {cl} ({prob})')


def main():
    '''Run the classification with all the options provided and return the predictions'''

    print('Validating arguments...')
    args = parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.gpu:
        device = torch.device("cpu")
    else:
        raise Exception("[ERROR] --gpu option: No GPU found")

    print('All arguments valid!\nQuerying the classifier...')
    predictions = predict(
        args.image_path,
        args.checkpoint,
        args.top_k,
        device
    )

    print('Query complete!\n Image results: ')
    print_pred(predictions, args.category_names)
    return predictions


if __name__ == '__main__':
    main()
