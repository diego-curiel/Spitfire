# Spitfire
## Description
The script takes a CSV file as input (typically a database dump) and splits the dataset into multiple DataFrames based on unique category values. 
Each DataFrame contains only the rows corresponding to a single category, preserving the original data order. 
Finally, it saves each DataFrame as a separate Excel file.
## Options
* -h, --help            show this help message and exit
* -i, --input INPUT     Path to the input file.
* -c, --category-column CATEGORY_COLUMN
                        Name of the column that contains the categories (case sensitive).
* -u, --uppercase       Set all strings inside the dataset to uppercase, including the file header.
## Function Example
### original.file.csv
| index |category|
|:-----:|:------:|
|  1    |    A   |
|  2    |    A   |
|  3    |    B   |
|  4    |    C   |
|  5    |    D   |
|  6    |    B   |
### Turns into
### original.file.split.xlsx (sheet A)
| index |category|
|:-----:|:------:|
|  1    |    A   |
|  2    |    A   |
### original.file.split.xlsx (sheet B)
| index |category|
|:-----:|:------:|
|  3    |    B   |
|  6    |    B   |
### original.file.split.xlsx (sheet C)
| index |category|
|:-----:|:------:|
|  4    |    C   |
### original.file.split.xlsx (sheet D)
| index |category|
|:-----:|:------:|
|  5    |    D   |
