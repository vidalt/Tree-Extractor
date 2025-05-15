# Load packages
import os as os
import argparse
from utils.experiments import (
    mainDT,
    mainRF,
    mainDTH,
    mainRFH,
)


def main(args):
    print("End of the experiment")
    if args.option == "RF":
        mainRF(args)
    elif args.option == "DT":
        mainDT(args)
    elif args.option == "DTH":
        mainDTH(args)
    elif args.option == "RFH":
        mainRFH(args)
    else:
        raise ValueError("Invalid option. Choose from RF, DT, DTH, RFH.")
    print("End of the experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Define the --experiment argument
    parser.add_argument(
        "--option",
        type=str,
        default="DT",
        required=True,
        help="The type of experiment to run (RF, DT, DTH, RFH).",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        required=True,
        help="The number of the experiment to run.",
    )

    # Parse the arguments
    args = parser.parse_args()
    main(args)
