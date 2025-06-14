import argparse
from predicting_failure.data_creation import create_train_data, create_test_data

def main(args):

    create_train_data(
        data_path=args.data_path,
        out_name=args.output_path,
        window_size=150,
        n_engines=args.n_engines,
    )

    create_test_data(
        args.data_path,
        args.true_rul_path, 
        out_name="test_engine_data.h5", 
        window_size=150, 
        n_engines=args.n_engines
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, required=True,
                        help="Path to the input data file.")
    parser.add_argument("-t", "--true_rul_path", dest="true_rul_path", type=str, required = True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str, default="train_engine_data.h5",
                        help="Path to save the output HDF5 file. Default is 'train_engine_data.h5'.")
    parser.add_argument("-n", "--n_engines", dest="n_engines", type=int, default=-1,
                        help="Number of engines to use from the input data. Default is -1 (use all engines).")
    args = parser.parse_args()
    main(args)
