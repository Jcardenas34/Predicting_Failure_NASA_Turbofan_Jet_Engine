from predicting_failure.helpers import load_data


def test_load_data():
    data_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/train_unit_all.h5"
    features, labels = load_data(data_path, 10)
    # print(f"Features shape: {features.shape}")
    # print(f"Labels shape: {labels.shape}")

    
    # print(f"Labels: {labels}")

    # for i in labels:
        # print(i[:10])
    
    # for i in features:
        # print(i[:1])

test_load_data()