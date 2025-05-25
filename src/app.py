#!/usr/bin/env python

import argparse as ap
import cProfile as pr
import math
import os
from pathlib import Path
from typing import Union

import pandas as pd

EXCEL_MAX_ROW = 1048576

def generate_arguments() -> ap.Namespace:
    """
    Generate the system arguments used in this script.
    
    Return
        ap.Namespace: Object that contains all arguments parsed.
    """
    parser = ap.ArgumentParser(
            prog="Spitfire",
            description="""
            Takes a CSV file as input (normally a database dump file), then 
            splits the dataset into multiple DataFrames based on unique category 
            values.  
            Each resulting DataFrame contains only the rows corresponding to a 
            single category, preserving the original order of the data.
            """
        )

    # File location
    parser.add_argument("-i", "--input", help="Path to the input file.",
                        required=True, type=Path)
    
    # Column to split
    # TODO: Group by multiple columns
    parser.add_argument("-c", "--category-column", 
                        help="""
                        Name of the column that contains the categories 
                        (case sensitive).
                        """,
                        required=True)
    
    parser.add_argument("-m", "--max-row", 
                        help="Set the max_row per sheet for the entire file",
                        default=EXCEL_MAX_ROW, type=int)

    parser.add_argument("-u", "--uppercase",
                        help="""
                        Set all strings inside the dataset to uppercase, 
                        including the file header.
                        """,
                        action="store_true", default=False)

    parser.add_argument("-p", "--prefix",
                        help="Add a prefix to the name of the output file/s.",
                        default="", type=str)

    parser.add_argument("-o", "--output", help="Path to the output file.",
                        required=True, type=Path)

    parser.add_argument("--separate", 
                        help="Save the groups into separated files each.",
                        action="store_true")

    return parser.parse_args()


def do_file_exist(file_path: Path, nonexistent_msg:str|None=None, 
               no_access_msg:str|None=None) -> bool:
    """
    Function to check if the filepath exist and it is available to be read.

    Params
        file_location (Path): The location to be asserted as accessible.
        nonexistent_msg (str): The message to be shown when the path doesn't exist.
        no_access_msg (str): The message to be shown when the file exist but it is
        not readable.

    Raise
        SystemExit = If the file does not exist.
        SystemExit = If the file exist but is not accessible (read mode).

    Return
        bool: True if both conditions passed.
    """

    dft_nonexistent_msg = "{f} does not exist!".format(f=file_path)
    
    # Check if the file exists
    if not file_path.exists(follow_symlinks=True):
        raise SystemExit(nonexistent_msg or dft_nonexistent_msg)
    
    dft_no_access_msg = "You do not have permissions to read {f}."

    dft_no_access_msg = dft_no_access_msg.format(f=file_path)

    # Check if the file is readable
    if not os.access(file_path, os.R_OK):
        raise SystemExit(no_access_msg or dft_no_access_msg)
    
    return True


def get_file_name(file_path:Path|str) -> str:
    """
    Get the name of the file without the extension.

    Arguments:
        file_path (Path|str): Path to the file location.

    Returns:
        str: File name without extension.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    filename = file_path.name
    filename, extension = os.path.splitext(filename)
    del extension

    return filename


def uppercase_dataframe(df:pd.DataFrame)->pd.DataFrame:
    """
    Takes a DataFrame and sets all of its textfields to uppercase.

    Parameters:
        df (pd.DataFrame): The DataFrame whose textfields will be set to upper.

    Returns:
        pd.Dataframe: A copy of the passed DataFrame with each textfield set to upper.
    """
    textfield_to_upper = lambda x: str(x).upper() if isinstance(x, str) else x

    df = df.map(textfield_to_upper)
    df.columns = [textfield_to_upper(x) for x in df.columns]

    return df


def split_dataset(
    df:pd.DataFrame, 
    category_column:str|int
) -> dict[Union[str, int], pd.DataFrame]: 
    """
    Splits the dataset into multiple DataFrames based on unique category values.

    Each resulting Dataframe contains only the rows corresponding to a single
    category, preserving the original order of the data. 

    Arguments:
        df: (pd.DataFrame): The DataFrame to be separated by its categories.
        category_column (str|int): The name of the category column.

    Return
        dict: Hashmap that contains the groups of DataFrames.
    """
    # Check if the category_column exist in the Input dataset
    if not category_column in df.columns:
        raise SystemExit(
            f"The column {category_column} does not exist in the input file!"
        )

    # Gather all categories available on this dataframe
    category_set = set(df.loc[:,category_column])

    # Filter the chunk by the values in the group_column
    groups = dict()
    for category in category_set:
        groups[category] = df[df[category_column] == category]

    return groups


def split_csv_file(
    input_path:Path|str, 
    category_column: Union[int, str],
) -> dict[Union[str, int], pd.DataFrame]:
    """
    Reads the file from the given input_path by chunks, then calls the
    split_dataset function for each chunk.
       
    Arguments:
        input_path (Path | str): Path to the dataset to be processed.
        category_column (int | str): Name of the column containing the categories.
       
    Returns:
        dict[str | int, pd.DataFrame]: Dictionary containing the groups.
    """
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    grouped_dataframes: dict[Union[str, int], pd.DataFrame] = dict()
    with pd.read_csv(input_path, chunksize=100000) as file:
        for chunk in file:
            chunk:pd.DataFrame

            # Loading animation
            print(".  ", end="\r")
            print(".. ", end="\r")
            print("...", end="\r")
            # ==================

            temporary_groups = split_dataset(
                df=chunk, 
                category_column=category_column
            )

            for category in temporary_groups.keys():
                group_df = grouped_dataframes.get(category, pd.DataFrame())
                temp_group = temporary_groups.get(category, pd.DataFrame())
                # Remove unwanted characters from the category such as '/'
                category = str(category).replace('/', '')
                grouped_dataframes[category] = pd.concat(
                    [group_df, temp_group],
                    ignore_index=True
                )

        # End of the loading animation
        print("DONE...", end="\n")
        # ==================

    return grouped_dataframes


def num_digits(n: int) -> int:
    """
    Get the amount of digits in a positive integer number.

    Parameters:
        n (int): the number whose amount of digits we want to know.

    Returns:
        int: The amount of digits in the given number.

    Raises:
        ValueError: If the number passed is negative.
    """
    if n < 0:
        raise ValueError("You cannot pass negative numbers")

    if n == 0:
        return 1

    return math.floor(math.log10(n)) + 1


def handle_sheet_overflow(df:pd.DataFrame, sheet_name:str, 
                          max_row:int) -> list[tuple[str | int, pd.DataFrame]]:
    """
    Splits a DataFrame into chunks based on the amount of rows
    passed on the max_row parameter, then it returns a list that contains a
    tuple for each chunk, whose first value is the new name of the sheet.

    E.G:
        max_row = 1024
        Sheet_A = 2100

        Sheet_A = 1024
        Sheet_A_2 = 1024
        Sheet_A_3 = 52

    Arguments:
        df (pd.DataFrame): Dataframe to split.
        sheet_name (str): Name of the sheet to be saved.
        max_row (int): Max amount of rows supported by the filetype.


    Returns:
        list[tuple]: List containing a tuple with the Sheet names and the DataFrames.
    """
    result = list()
    total_rows = len(df)
    num_chunks = math.ceil(total_rows / max_row)
    pad_width = num_digits(num_chunks)

    for sheet_count, start in enumerate(range(0, total_rows, max_row), start=1):
        end = start + max_row
        # Suffix example: _001, _002
        suffix = f"_{str(sheet_count).rjust(pad_width, "0")}"
        # Excel has a limit of 31 characters for the sheet name
        max_base_len = 31 - len(suffix)
        trimmed_name = sheet_name[:max_base_len].rstrip("_ ")
        final_name = f"{trimmed_name}{suffix}"
        result.append((final_name, df.iloc[start:end]))

    return result


def main():
    sys_args = generate_arguments()
    # Check the input file accessibility
    do_file_exist(file_path=sys_args.input)

    output_file_name = get_file_name(sys_args.output)

    # Validate the save directory
    if not sys_args.output.parent.exists():
        raise SystemExit("The save directory does not exist!")

    # Validate the max-row argument
    if sys_args.max_row > EXCEL_MAX_ROW:
        sys_err = "The maximum supported range for rows is: {max_rows}"
        sys_err = sys_err.format(max_rows=EXCEL_MAX_ROW)
        raise SystemExit(sys_err)

    # Set profiler to create the performance file
    with pr.Profile() as profiler:
        # Save the groups into Excel File
        print("Reading file...")

        grouped_dataframes = split_csv_file(
            input_path=sys_args.input,
            category_column=sys_args.category_column,
        )

        if sys_args.uppercase:
            print("The files will be saved with textfields in uppercase")
            input("Press any key to continue...")

            grouped_dataframes = {str(k).upper():uppercase_dataframe(v)
                                  for k,v in grouped_dataframes.items()}

        # Template for the output file path
        output_template = "{parent}/{prefix}{category}.{suffix}.xlsx"

        # Handle sheet overflow for each group
        sheets_per_group = {
            category:handle_sheet_overflow(
                df=df,
                sheet_name=str(category),
                max_row=sys_args.max_row,
            )
            for category, df in grouped_dataframes.items()
        }

        # This loop is for saving each group into its corresponding Excel files

        
        if sys_args.separate:
            # Save the groups separately
            for category, sheet_tuples in sheets_per_group.items():
                print(f"saving the file for the category {category}")
                output_path = output_template.format(
                    parent=Path(sys_args.output).parent,
                    prefix=sys_args.prefix,
                    category=category,
                    suffix=output_file_name,
                )
                # Save the corresponding sheets to each separated file
                with pd.ExcelWriter(output_path) as excel_writer:
                    for sheet_name, content in sheet_tuples:
                        content.to_excel(excel_writer, index=False,
                                         sheet_name=str(sheet_name))

        # Save the groups into one file
        else:
            output_path = output_template.format(
                parent=Path(sys_args.output).parent,
                prefix= sys_args.prefix,
                category=output_file_name,
                suffix="split",
            )
            print(f"Saving the file {output_path}")
            with pd.ExcelWriter(output_path) as excel_writer:
                for _, sheet_tuples in sheets_per_group.items():
                    for sheet_name, content in sheet_tuples:
                        content: pd.DataFrame
                        content.to_excel(excel_writer, index=False, 
                                        sheet_name=str(sheet_name))

        # Gather performance information with cProfile
        performance_info = Path(sys_args.output).parent.joinpath("profiler_report")
        profiler.create_stats()
        profiler.dump_stats(performance_info)


if __name__ == "__main__":
    main()
