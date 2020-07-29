#!/usr/bin/env python3
import pandas as pd

"""
python read/write excel or csv file
"""


def read_csv(file, encoding):
    data_df = pd.read_csv(file, encoding=encoding)
    return data_df


def write_csv(file, data_df, encoding):
    data_df.to_csv(file, encoding=encoding)


def read_excel(file, sheet_name):
    """
    read excel file
    :param file:
    :param sheet_name:
    :return:
    """
    reader = pd.ExcelFile(file)
    data_df = reader.parse(sheet_name)
    return data_df


def write_excel(file, sheet_name, data_df):
    """
    write excel file
    :param file:
    :param sheet_name:
    :return:
    """
    writer = pd.ExcelWriter(file)
    data_df.to_excel(writer, sheet_name=sheet_name, index=False)
