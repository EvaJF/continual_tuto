import argparse
import pprint


def parse_args():
    pp = pprint.PrettyPrinter()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name, for instance 'mnist', 'flowers-102', 'food-101' ",
    )
    parser.add_argument(
        "--nb_init_cl",
        type=int,
        default=500,
        help="number of (real) initial classes / B",
    )
    parser.add_argument(
        "--nb_incr_cl",
        type=int,
        default=50,
        help="number of (real) initial classes / B",
    )
    parser.add_argument(
        "--nb_tot_cl", type=int, default=1000, help="total number of (real) classes"
    )
    parser.add_argument(
        "--archi",
        type=str,
        default="resnet18",
        choices=["resnet18", "vits", "simpleCNN"],
        help="architecture of the encoder",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="in1k",
        choices=["in1k", "in21k", "none", "lvd142m"],
        help="pretraining dataset for the encoder",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="path to features directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="folder for saving results"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="debug",
        help="model identifier for the features subfolder",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=10.0,
        help="coeff1 for covariance shrinkage in fecam",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=10.0,
        help="coeff2 for covariance shrinkage in fecam",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=0, help="new feature size for random projection"
    )

    args = parser.parse_args()
    pp.pprint(args)

    return args
