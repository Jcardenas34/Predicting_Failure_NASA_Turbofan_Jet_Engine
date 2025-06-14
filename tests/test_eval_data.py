import torch.nn as nn
from predicting_failure.helpers import load_eval_data, load_data
from predicting_failure.core_train_models import evaluate_model
import argparse

def test_evaluate_model(args):

    # data_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_test_8.h5"
    # model_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/models/RUL_regressor_unit1_epoch_9.pth"

    eval_loader = load_eval_data(args.data_path)
    loss_function = nn.L1Loss()

    evaluate_model(model_path=args.model_path, data_path=args.data_path, eval_loader=eval_loader, loss_function=loss_function)


def test_load_data():

    data_path_train = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_1.h5"
    data_path_test = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_test_1.h5"
    model_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/models/RUL_regressor_unit1_epoch_9.pth"

    train_loader, val_loader = load_data(data_path_train)
    eval_loader = load_eval_data(data_path_test)

    return eval_loader

# test_load_data()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_path", type=str, required=True)
    parser.add_argument("-m","--model_path", type=str, required=True)
    args = parser.parse_args()
    
    test_evaluate_model(args)