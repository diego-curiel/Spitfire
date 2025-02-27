import pandas as pd

import argparse as ap
from pathlib import Path

def generate_arguments()->ap.Namespace:
    parser = ap.ArgumentParser(
        prog="Excel to CSV",
        description="Fast hack to turn a XLSX into CSV"
    )

    parser.add_argument("-i", "--input", help="Input XLSX file path", type=Path)
    parser.add_argument("-o", "--output", help="Output CSV file path",
                        type=Path)
    return parser.parse_args()

def main():
    SYS_ARGS = generate_arguments()

    input_df = pd.read_excel(SYS_ARGS.input)

    input_df.to_csv(SYS_ARGS.output, index=False)

if __name__ == "__main__":
    main()

