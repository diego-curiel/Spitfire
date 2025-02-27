import pandas as pd

import argparse as ap
import cProfile as pr
import os
from pathlib import Path
from typing import Any

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
    parser.add_argument("-o", "--output", help="Path to the output file.",
                        required=True, type=Path)
    parser.add_argument("--separate", 
                        help="Save the groups into separated files each.",
                        action="store_true")

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
    
    dft_no_access_msg = "You do not have permissions to read {f}."

    dft_no_access_msg = dft_no_access_msg.format(f=file_path)

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
    filename = file_path.name
    filename, extension = os.path.splitext(filename)
    del extension

    return filename


def dataframe_upercase(df:pd.DataFrame)->pd.DataFrame:
    """Description:
    Takes a DataFrame and sets all of its textfields to uppercase.

    Parameters:
    df (pd.DataFrame) - The DataFrame whose textfields will be set to upper.

    Returns:
    pd.Dataframe - A copy of the passed DataFrame with each textfield set to upper.
    """
    textfield_to_upper = lambda x: str(x).upper() if isinstance(x, str) else x

    df = df.map(textfield_to_upper)
    df.columns = [textfield_to_upper for x in df.columns]

    return df


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
    categories = set(df.loc[:,category_column])

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
                          max_row:int) -> list[tuple]:
    """Description:
    This function splits a DataFrame into chunks based on the amount of rows
    passed on the max_row parameter, then it returns a list that contains a
    tuple for each chunk, whose first value is the new name of the sheet.
    E.G:
    max_row = 1024
    Sheet_A = 2100

    Sheet_A = 1024
    Sheet_A_2 = 1024
    Sheet_A_3 = 52

    Arguments:
    df (pd.DataFrame) - Dataframe to split.
    sheet_name (str) - Name of the sheet to be saved.
    max_row (int) - Max amount of rows supported by the filetype.

    Returns:
    list[tuple] - list containing a tuple with the Sheet names and the DataFrames.
    """
    # Initialize the variables
    result = list()
    index = 1
    new_df = df.copy()
    new_name = sheet_name 
    #Do while
    while True:
        # Stop when there's no splitting left to do
        if len(new_df) == 0:
            break

        # Create a new sheet_name
        if index > 1:
            new_name = "{sn}_{i}".format(sn=sheet_name, i=index)

        # Append the split DataFrame to the result list
        sheet_tuple:tuple[str, pd.DataFrame] = (new_name, new_df.iloc[:max_row])
        new_df:pd.DataFrame = new_df.iloc[max_row:]
        result.append(sheet_tuple)
        # Increase the index
        index += 1
    
    return result


def main():
    SYS_ARGS = generate_arguments()
    
    # Input file information
    INPUT_PATH:Path = SYS_ARGS.input
    INPUT_DIR = INPUT_PATH.parent
    # Check file accessibility
    file_exist(file_path=INPUT_PATH)
    
    # Output file information
    OUTPUT_PATH:Path = SYS_ARGS.output
    OUTPUT_DIR = OUTPUT_PATH.parent
    OUTPUT_FILENAME = get_file_name(OUTPUT_PATH)
   
    # Max rows per sheet
    MAX_ROW = SYS_ARGS.max_row

    # If the save file does not exist, raise an exception
    if not OUTPUT_PATH.parent.exists():
        raise SystemExit("The save directory does not exist!")

    # Handle bad max_row inputs
    if SYS_ARGS.max_row > EXCEL_MAX_ROW:
        sys_err = "The maximum supported range for rows is: {r}"
        sys_err = sys_err.format(r=EXCEL_MAX_ROW)
        raise SystemExit(sys_err)

    # Set profiler to create the performance file
    with pr.Profile() as profiler:
        # Yield the grouped DataFrames
        grouped_dataframes = grouped_dataframes_generator(
                input_path=INPUT_PATH,
                sys_args=SYS_ARGS
            )
        
        # Small function to set text to uppercase ignoring other types
        to_upper = lambda x: x.upper() if isinstance(x, str) else x

        if SYS_ARGS.uppercase:
            print("The files will be saved with textfields in uppercase")
            input("Press any key to continue...")

        # Save the groups into Excel File
        print("Reading file...")
        # Option to save the groups into separated files each.
        if SYS_ARGS.separate:
            for category, df in grouped_dataframes:
                # Check if the user wants their spreadsheet all in uppercase
                if SYS_ARGS.uppercase:
                    df:pd.DataFrame = dataframe_upercase(df)
                    category = to_upper(category)

                # Change file name according to the category
                output_file = "{parent}/{filename}.{category}.xlsx"
                output_file = output_file.format(parent=OUTPUT_DIR,
                                                 filename=OUTPUT_FILENAME,
                                                 category=category)

                with pd.ExcelWriter(output_file, mode="w") as excel_writer:
                    sheet_tuple = handle_sheet_overflow(df=df,
                                                        sheet_name=category,
                                                        max_row=MAX_ROW)
                    print_msg = "Saving group {group_name}..."
                    print_msg = print_msg.format(group_name=category)
                    print(print_msg)

                    for sheet_name, df in sheet_tuple:
                        df.to_excel(excel_writer, index=False, 
                                    sheet_name=sheet_name)

        else:
            with pd.ExcelWriter(OUTPUT_PATH, mode="w") as excel_writer:
                for category, df in grouped_dataframes:
                    # Check if the user wants their spreadsheet all in uppercase
                    if SYS_ARGS.uppercase:
                        df:pd.DataFrame = dataframe_upercase(df)
                        category = to_upper(category)

                    sheet_tuple = handle_sheet_overflow(df=df,
                                                        sheet_name=category,
                                                        max_row=MAX_ROW)
                    print_msg = "Saving group {group_name}..."
                    print_msg = print_msg.format(group_name=category)
                    print(print_msg)

                    for sheet_name, df in sheet_tuple:
                        df.to_excel(excel_writer, index=False,
                                    sheet_name=sheet_name)

        # Gather performance information with cProfile
        performance_info = INPUT_DIR.joinpath("profiler_report")
        profiler.create_stats()
        profiler.dump_stats(performance_info)
    

if __name__ == "__main__":
    main()
