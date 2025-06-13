from predicting_failure.helpers import load_data


def test_load_data():
    data_path = "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_1.h5"
    features, labels = load_data(data_path)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    
    # print(f"Labels: {labels}")

    # for i in labels:
        # print(i[:10])
    
    # for i in features:
        # print(i[:1])

test_load_data()