import argparse
import torch.nn as nn
from predicting_failure.helpers import load_eval_data_singleRUL
from predicting_failure.core_train_models import evaluate_model


def main(args):

    eval_loader = load_eval_data_singleRUL(data_path=args.data_path, 
                                           scalar_path="scalars/scalar_FD001.pkl",
                                           batch_size=32, 
                                           n_samples=args.n_samples
                                           )
    loss_function = nn.L1Loss()

    evaluate_model(model_path=args.model_path, data_path=args.data_path, eval_loader=eval_loader, loss_function=loss_function)



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_path", type=str, required=True)
    parser.add_argument("-m","--model_path", type=str, required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=-1, help="Number of samples to evaluate. Default is -1 (use all samples).")
    args = parser.parse_args()
    
    main(args)