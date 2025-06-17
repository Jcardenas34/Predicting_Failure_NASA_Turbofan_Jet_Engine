from predicting_failure.helpers import load_data


def test_load_data():
    data_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/train_engine_data.h5"
    features, labels = load_data(data_path, 10)


test_load_data()