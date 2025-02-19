import pandas as pd

import argparse as ap
import cProfile as pr
import os
from pathlib import Path
from typing import Any
import sys

EXCEL_MAX_ROW = 1048576

def generate_arguments() -> ap.Namespace:
    """
    Description
    -----------
    Generate the system arguments used in this script.
    --------
    Return
    --------
    ap.Namespace = Object that contains all arguments parsed.
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

    return parser.parse_args()


def file_exist(file_path: Path, nonexistent_msg:str|None=None, 
               no_access_msg:str|None=None) -> None:
    """
    Description
    -----------
    Function to check if the filepath exist and it is available to be read.
    ------
    Params
    ------
    file_location:Path = The location to be asserted as accessible.
    nonexistent_msg:str = The message to be shown when the path doesn't exist.
    no_access_msg:str = The message to be shown when the file exist but it is
    not readable.
    -----
    Raise
    -----
    SystemExit = If the file does not exist.
    SystemExit = If the file exist but is not accessible (read mode).
    ------
    Return
    ------
    None
    """

    dft_nonexistent_msg = "{f} does not exist!".format(f=file_path)
    
    # Check if the file exists
    if not file_path.exists(follow_symlinks=True):
        raise SystemExit(nonexistent_msg or dft_nonexistent_msg)
    
    dft_no_access_msg = "You do not have permissions to read {f}".format(
            f=file_path
        )

    # Check if the file is readable
    if not os.access(file_path, os.R_OK):
        raise SystemExit(no_access_msg or dft_no_access_msg)


def get_file_name(file_path:Path|str) -> str:
    """
    Description
    -----------
    Get the name of the file without the extension.
    -----
    Args
    -----
    file_path:Path|str = path to the file location.
    ------
    Return
    ------
    str = file name without extension.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    return ".".join(file_path.name.split(".")[:-1])


def split_dataset(df:pd.DataFrame, 
                  category_column:str|int) -> dict[Any, pd.DataFrame]: 
    """
    Description
    ------------
    Splits the dataset into multiple DataFrames based on unique category values.
    Each resulting Dataframe contains only the rows corresponding to a single
    category, preserving the original order of the data. 
    ----
    Args
    ----
    df:pd.DataFrame = Input dataset.
    category_column:str|int = The horizontal index where the groups are located.
    ------
    Return
    -------
    dict = Hashmap that contains the groups of DataFrames.
    """
    # Check if the category_column exist in the Input dataset
    if not category_column in df.columns:
        raise SystemExit(
                "The column {c} does not exist in the input file!".format(
                        c=category_column
                    )
            )
    
    # Gather all categories available on this dataframe
    categories = set(df[category_column])

    # Filter the chunk by the values in the group_column
    groups = dict()
    for category in categories:
        groups[category] = df[df[category_column] == category]
            
    return groups


def grouped_dataframes_generator(input_path:Path|str, 
                                sys_args:ap.Namespace):
    """
    Description
    ---
    This function takes a Dataset and applies split_dataset on it while saving
    memory (Lazy loading). All of the groups are then yielded as a tuple that 
    contains the category value and the DataFrame itself.
    ---
    Args
    ---
    input_path:Path|str = Path to the dataset to be processed.
    sys_args:ap.Namespace = System Arguments.
    ---
    Yields
    ---
    tuple[str, pd.DataFrame] = (category, DataFrame)
    """
    if not isinstance(input_path, Path):
        input_path = Path(input_path)


    ultimate_groups = dict()
    with pd.read_csv(input_path, chunksize=100000) as file:
        # List of files to turn into Excel later
        # TODO: Add options for files longer than the max row count of Excel

        # Chunk processing loop
        for chunk in file:
            print(".  ", end="\r")
            chunk:pd.DataFrame
            temporary_groups = split_dataset(df=chunk, 
                                      category_column=sys_args.category_column)
            print(".. ", end="\r")
            
            for category in temporary_groups.keys():
                ult_group = ultimate_groups.get(category, pd.DataFrame())
                temp_group = temporary_groups.get(category, pd.DataFrame())

                ultimate_groups[category] = pd.concat([ult_group, temp_group],
                                                      ignore_index=True)

            print("...", end="\r")
        print("DONE...", end="\n")

    for value in tuple(ultimate_groups.items()):
        yield value


def handle_sheet_overflow(df:pd.DataFrame, sheet_name:str,
                    max_row:int, index:int=1, 
                    result:list|None=None) -> list:
    """
    Description
    -----------
    This function takes a DataFrame and counts its rows. If the DataFrame rows 
    are less or equal than the max_row value then it returns a dictionary with 
    the sheet name as the key, and the DataFrame as value. Otherwise, if the 
    DataFrame row count is bigger than the max_row value, then the function will
    recursively split the DataFrame by the max_row value, adding a numeral to
    the sheet name. E.G:
        max_row = 1024
        Sheet_A = 2100

        Sheet_A = 1024
        Sheet_A_2 = 1024
        Sheet_A_3 = 52
    ---
    Arguments
    ---------
    df:pd.DataFrame = Dataframe to split.
    sheet_name: Name of the sheet to be saved.
    max_row: Max amount of rows supported by the filetype.
    index:int = Iteration index number.
    result:list|None (default=None) = Result of the past iteration.
    ---
    Return
    ------
    list = list containing a tuple with the Sheet names and the DataFrames.
    """
    # Create an empty list to save results
    if not isinstance(result, list):
        result = list()
    # Stop when there's no splitting left to do
    if len(df) == 0:
        return result
    # Create a new sheet_name
    new_name = sheet_name
    if index > 1:
        new_name = "{sn}_{i}".format(sn=sheet_name, i=index)
    # Append the split DataFrame to the result list
    result.append((new_name, df.iloc[:max_row]))
    # Search again
    return handle_sheet_overflow(df=df.iloc[max_row:], sheet_name=sheet_name,
                           max_row=max_row, index=index + 1,
                           result=result)


def main():
    sys_args = generate_arguments()

    file_path = Path(sys_args.input)
    
    # Check file accessibility
    file_exist(file_path=file_path)

    parent_directory = file_path.parent

    file_name = get_file_name(file_path)
    
    OUTPUT_FILE = parent_directory.joinpath(
            "{f}.split.xlsx".format(f=file_name)
        )
    RECURSION_DEPTH = sys.getrecursionlimit()

    # Handle bad max_row inputs
    if sys_args.max_row > EXCEL_MAX_ROW:
        sys_err = "The maximum supported range for rows is: {r}".format(
                r=EXCEL_MAX_ROW
            )
        raise SystemExit(sys_err)
    # Set profiler to create the performance file
    with pr.Profile() as profiler:
        # Yield the grouped DataFrames
        grouped_dataframes = grouped_dataframes_generator(
                input_path=file_path,
                sys_args=sys_args
            )

        if sys_args.uppercase:
            print("The files will be saved with textfields in uppercase")
            input("Press any key to continue...")

        # Save the groups into Excel File
        with pd.ExcelWriter(
                OUTPUT_FILE, mode="w", engine="xlsxwriter") as writer:
            print("Reading file...")
            for group in grouped_dataframes:
                category, df = group
                df:pd.DataFrame

                # Ensure that the category is a string
                if not isinstance(category, str):
                    category = str(category)

                # If needed, set all of the dataset text fields to uppercase
                if sys_args.uppercase:
                    to_upper = lambda x: str(x).upper() if isinstance(x, str) else x
                    df = df.map(to_upper)
                    df.columns = [to_upper(x) for x in df.columns]
                    category = to_upper(category)

                print("saving group {c}...".format(c=category))
                print("group size: {s}".format(s=len(df)))
                
                # Fix to handle stack overflow
                max_row = sys_args.max_row
                splits = len(df) // max_row

                if RECURSION_DEPTH < splits:
                    # minus two iterations
                    sys_err = "ERROR!, the row count for {c} is too low!"
                    sys_err_ex = " Try increasing the --max-row option value."
                    sys_err = sys_err.format(c=category) + sys_err_ex
                    raise SystemExit(sys_err)

                df_split = handle_sheet_overflow(df=df, sheet_name=category,
                                                 max_row=max_row)

                for sheet_name, split_sheet in df_split:
                    split_sheet.to_excel(writer, index=False, 
                                         sheet_name=sheet_name)

        # Gather performance information with cProfile
        performance_info = parent_directory.joinpath("profiler_report")
        profiler.create_stats()
        profiler.dump_stats(performance_info)
    

if __name__ == "__main__":
    main()
