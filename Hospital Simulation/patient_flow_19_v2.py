from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Creating 2019 Patient Flow
# Read the patient flow data from 'sheet1' of 'flow' Excel file using 'openpyxl' engine
df_sheet1 = pd.read_excel('flow.xlsx', sheet_name='sheet1', engine='openpyxl')

quarters = df_sheet1.columns[1:4]
quarter_start_dates = df_sheet1.iloc[2, 1:4].tolist()
quarter_end_dates = df_sheet1.iloc[3, 1:4].tolist()
daily_means = df_sheet1.iloc[0, 1:4].tolist()
daily_stds = df_sheet1.iloc[1, 1:4].tolist()

# specify timeframe for first year of patient flow [2019]
start_date = datetime(2019, 1, 1, 0, 59, 0)
end_date = datetime(2019, 12, 31, 23, 59, 0)
num_hours = int((end_date - start_date).total_seconds() / 3600) + 1

# Generate time series to show all hours of patient flow
datetime_list = [start_date + timedelta(hours=i) for i in range(num_hours)]

# Initialize an empty list to store the generated patient visits for each hour
patient_visits = []

# Loop through each datetime in datetime_list
for dt in datetime_list:
    quarter_index = 0
    # Find the corresponding quarter for the current datetime
    for i in range(len(quarter_start_dates)):
        if quarter_start_dates[i] <= dt <= quarter_end_dates[i]:
            quarter_index = i
            break
    # Generate a patient visit volume list based on the quarterly daily mean and standard deviation in 2019
    daily_mean = daily_means[quarter_index]
    daily_std = daily_stds[quarter_index]
    num_visits = int(np.random.normal(daily_mean, daily_std))
    # Append the patient visits for the current hour to the patient_visits list
    patient_visits.append(num_visits)

# Create a DataFrame with datetime and patient_visits columns
df_patient_visits = pd.DataFrame({'datetime': datetime_list, 'patient_visits': patient_visits})

# Save the DataFrame to a CSV file
df_patient_visits.to_csv('2019_patient_visits.csv', index=False)

# STEP 2

# Initialize empty lists to store patient types and visit times for each row
patient_type_lists = []
visit_time_lists = []

# Loop through each row in the DataFrame
for index, row in df_patient_visits.iterrows():
    num_visits = row['patient_visits']
    patient_types = ['A', 'B', 'C', 'D']
    weights = [0.30, 0.37, 0.20, 0.13]
    patient_types = np.random.choice(patient_types, size=num_visits, p=weights)
    visit_times = [row['datetime'] - timedelta(minutes=np.random.randint(1, 60)) for _ in range(num_visits)]
    patient_type_lists.append(patient_types)
    visit_time_lists.append(visit_times)

# Determine the maximum number of patient visits across all rows
max_patient_visits = max(df_patient_visits['patient_visits'])

# Add patient_type_lists and visit_time_lists as new columns to the DataFrame
df_patient_visits['patient_types'] = patient_type_lists
df_patient_visits['visit_times'] = visit_time_lists

# Create new columns for each patient type and visit time
for i in range(max_patient_visits):
    df_patient_visits[f'patient_type_{i + 1}'] = [ptype[i] if i < len(ptype) else np.nan for ptype in
                                                  patient_type_lists]
    df_patient_visits[f'visit_time_{i + 1}'] = [vtime[i].strftime('%Y-%m-%d %H:%M:%S') if i < len(vtime) else np.nan for
                                                vtime in visit_time_lists]

# Sort the DataFrame by 'datetime' column
df_patient_visits.sort_values(by='datetime', inplace=True)

# Save the DataFrame to a CSV file
df_patient_visits.to_csv('2019_patient_visits2.csv', index=False)

# STEP 3: Create final flow for 2019

# Read the patient_visits2.csv file
df_patient_visits = pd.read_csv('2019_patient_visits2.csv')

# Initialize an empty list to store patient records
patient_records = []

# Loop through each row in the hourly patient volume DataFrame
for _, row in df_patient_visits.iterrows():
    # Loop through each patient type and visit time
    for i in range(1, row['patient_visits'] + 1):
        patient_type_col = f'patient_type_{i}'
        visit_time_col = f'visit_time_{i}'
        patient_type_val = row[patient_type_col]
        visit_time_val = row[visit_time_col]
        # Append a patient record as a tuple (visit_time_val, patient_type) to the patient_records list
        patient_records.append((visit_time_val, patient_type_val))

# Create a new DataFrame from the patient_records list
df_final_2019 = pd.DataFrame(patient_records, columns=['datetime', 'patient_type'])

# Add a new column 'patient_id' containing unique patient IDs
df_final_2019['patient_id'] = range(1, len(df_final_2019) + 1)

# Sort the DataFrame by 'datetime' column to ensure it is in chronological order
df_final_2019.sort_values(by='datetime', inplace=True)

# Save the DataFrame to a CSV file named 'final_2019.csv'
df_final_2019.to_csv('final_2019.csv', index=False)

# STEP 4: Create class for Patient19

class Patient19():
  def __init__(self, datetime_str, patient_type, patient_id):
    self.datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    self.patient_type = str(patient_type)
    self.patient_id = patient_id

def read_Patient19_data(filename):
    patient19 = []
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        patient19.append(Patient19(
            str(row['datetime']),
            str(row['patient_type']),
            int(row['patient_id']))
        )
    return patient19

final_2019_data = read_Patient19_data('final_2019.csv')