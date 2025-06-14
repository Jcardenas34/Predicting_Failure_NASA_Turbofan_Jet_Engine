coreimport torch.nn as nn
from predicting_failure.helpers import load_eval_data, load_data
from predicting_failure.core_train_models import evaluate_model
import argparse


def main(args):

    eval_loader = load_eval_data(args.data_path, args.n_samples)
    loss_function = nn.L1Loss()

    evaluate_model(model_path=args.model_path, data_path=args.data_path, eval_loader=eval_loader, loss_function=loss_function)



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_path", type=str, required=True)
    parser.add_argument("-m","--model_path", type=str, required=True)
    args = parser.parse_args()
    
    main(args)