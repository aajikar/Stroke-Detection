# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:12:05 2021

@author: parth
"""

from pathlib import Path         # Path class from pathlib library to handle path management
import numpy as np               # Numpy library handles arrays, vectors and matrix operations
import pandas as pd              # The pandas library gives is used to load the csv files and handle the data
from datetime import datetime
import functools
import time

"""
Group the data.
Assign label such that sequence is equal to self.seq_len.

New rows will be added to meet the desired group length. Patient ID, 
Frame Number and Seq ID will need to be included in those rows.

Labels will be added to a new column in the metadata dataframe.
Fine sequence labels will have metadata of pressure frames which belong
to the same patient and are part of a continguous sequence of data.

Returns
-------
None.

"""

def timer(func):
    """Print the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = start_time - end_time
        print(f"Finished {func.__name__!r} in {-run_time:.4f} s")
        return value
    return wrapper_timer

# Function to insert row in the dataframe
def Insert_row(row_number, df, patient_id_pre, seq_id_pre, frame_num, datetime, cop, 
               avg_pres, peak_pres, contact_area, filename, label_num, no_patient, garbage):
   
    # Insert a row at the end
    df.loc[row_number, 'Patient ID'] = patient_id_pre
    df.loc[row_number, 'Seq ID'] = seq_id_pre
    df.loc[row_number, 'Frame'] = frame_num
    df.loc[row_number, 'Datetime'] = datetime
    df.loc[row_number, 'COP'] = cop
    df.loc[row_number, 'Avg Pres'] = avg_pres
    df.loc[row_number, 'Peak Pres'] = peak_pres
    df.loc[row_number, 'Contact Area'] = contact_area
    df.loc[row_number, 'Filename'] = filename
    df.loc[row_number, 'Label'] = label_num
    df.loc[row_number, 'No Patient'] = no_patient
    df.loc[row_number, 'Garbage'] = garbage 
    
    return df

# Function to get data
def Frame_num(index):
    f_num = df.loc[index, 'Frame']
    return f_num

def Datetime_num(index):
    datetime = df.loc[index, 'Datetime']
    return datetime

def COP_num(index):
    cop = df.loc[index, 'COP']
    return cop

def Avg_pres_num(index):
    avg_pres = df.loc[index, 'Avg Pres']
    return avg_pres

def Peak_pres_num(index):
    peak_pres = df.loc[index, 'Peak Pres']
    return peak_pres

def Contact_area_num(index):
    contact_area = df.loc[index, 'Contact Area']
    return contact_area

def Fil_name_num(index):
    fil_name = df.loc[index, 'Filename']
    return fil_name

def Lab_num(index):
    lab = df.loc[index, 'Label']
    return lab

def No_patient_num(index):
    no_patient = df.loc[index, 'No Patient']
    return no_patient

def Garbage_num(index):
    garbage = df.loc[index, 'Garbage']
    return garbage

@timer
# def for lines of data associated with a Patient ID and grouping them to find the number of rows required to be added to complete the grouping
def Patient_ID_group(seq_len, num_data):
    
    required = num_data % seq_len                 # number of rows required to be added to complete 
                                                  # the last group of that Patient 
    return required

@timer
# def that removes seqs that have less than 1/2 of the specicifed seq_len
def Delete_(seq_len, required_rows, num_data):
    
    if required_rows > (seq_len/2):
        row_index = num_data - required_rows
        print (row_index)
        dropped = df.drop(df.index[row_index : num_data])
        return dropped
        
    else:
        return 2

@timer
def Adding_rows(patient_id_pre, seq_id_pre, required_rows, num_data, df):
    
    index = num_data - 1
    for k in range (required_rows + 2):
        frame_num = Frame_num(index)
        datetime = Datetime_num(index)
        cop = COP_num(index)
        avg_pres = Avg_pres_num(index)
        peak_pres = Peak_pres_num(index)
        contact_area = Contact_area_num(index)
        filename = Fil_name_num(index)
        label_num = Lab_num(index)
        no_patient = No_patient_num(index)
        garbage = Garbage_num(index)
        df = Insert_row(num_data, df, patient_id_pre, seq_id_pre, frame_num, datetime, cop, 
                        avg_pres, peak_pres, contact_area, filename, label_num, no_patient, garbage)
        index -= 1
        num_data += 1
        
    return df

# Add Fine Seq ID column
def Add_fine_seq_ID(seq_len, df):
    
    count_rows = 0
    label = 0
    fine_seq_id = []
    for row_name, dataframe in df.groupby(['Patient ID', 'Seq ID']):
        for _ in range (len(dataframe)):
            if (count_rows % seq_len == 0) and (count_rows != 0):
                label += 1
            fine_seq_id.append(label)
            count_rows += 1
        count_rows = 0
        label += 1
    
    df['Fine Seq ID'] = np.array(fine_seq_id).squeeze() # col vector adding to dataframe as fine seq id
    return df

def Implementation(length, seq_len, df):
    
    for i in range(length):
        num_data = (df.groupby(['Patient ID'])['Frame'].count().iloc[i])
        required_rows = Patient_ID_group(seq_len, num_data) # number of rows required to be added to complete the grouping 
        remove_rows_cols = Delete_(seq_len, required_rows, num_data) # removes seqs that have less than 1/2 of the specicifed seq_len
    
        if remove_rows_cols == 0:
            patient_id_pre = df.loc[(num_data - 1), 'Patient ID']
            seq_id_pre = df.loc[(num_data - 1), 'Seq ID']
            add_rows = Adding_rows(patient_id_pre, seq_id_pre, required_rows, num_data, df) # add the required rows to complete the group

@timer
def main_():
    
    seq_len = 60
    length = len(df.groupby(['Patient ID'])['Frame'].count())
    
    Implementation(length, seq_len, df)
    
    for i in range(length):
        num_data = (df.groupby(['Patient ID'])['Frame'].count().iloc[i])
        required_rows = Patient_ID_group(seq_len, num_data) # number of rows required to be added to complete the grouping 
        remove_rows_cols = Delete_(seq_len, required_rows, num_data) # removes seqs that have less than 1/2 of the specicifed seq_len
    
        if remove_rows_cols == 2:
            patient_id_pre = df.loc[(num_data - 1), 'Patient ID']
            seq_id_pre = df.loc[(num_data - 1), 'Seq ID']
            add_rows = Adding_rows(patient_id_pre, seq_id_pre, required_rows, num_data, df) # add the required rows to complete the group

root_path = Path(r'C:\Research2021\Data - Patients\Sample Data from Patients\p001\metadata.csv')
df = pd.read_csv(root_path)
main_()