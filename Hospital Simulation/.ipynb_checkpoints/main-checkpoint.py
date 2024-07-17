import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime, timedelta
import simpy
import random
import gzip
import numpy as np

# referencing modular script in same path as this script
# functions to generate a .xlsx file with series of patient
# arrival times (i.e. the patient dataset) across this
# simulation's one year timeline

from patient_flow_19_v2 import Patient19

# Load patient dataset as data_2019
data_2019 = pd.read_csv("final_2019.csv", parse_dates=['datetime'])

# store dataset as dataframe
data_2019 = pd.DataFrame(data_2019)

# create series of additional columns for dataframe
# each column will function to store timestamps across
# each unique patient's journey in the patient triage lifecycle
# e.g. time of arrival, checkin wait time, nurse consult time
# where each timestamp across columns represents the start time
# of the represented step in patient triage lifecycle

# Create new columns for timestamps in the dataframe to show process durations
# data_2019['Arrival_Time'] = data_2019['datetime'].apply(lambda dt: pd.Timestamp(dt).time())
data_2019['Wait_Time'] = np.NaN #pd.NaT
data_2019['Checkin_Time'] = np.NaN #pd.NaT
data_2019['Nurse_Consult_Time'] = np.NaN #pd.NaT
data_2019['Doctor_Consult_Time'] = np.NaN #pd.NaT
data_2019['Technician_Scan_Time'] = np.NaN #pd.NaT
data_2019['Nurse_Injection_Time'] = np.NaN #pd.NaT
data_2019['Nurse_Surgery_Time'] = np.NaN #pd.NaT
data_2019['Doctor_Surgery_Time'] = np.NaN #pd.NaT
data_2019['Surgery_Dress_Time'] = np.NaN #pd.NaT
data_2019['Surgery_Recover_Time'] = np.NaN #pd.NaT
data_2019['Checkout_Time'] = np.NaN #pd.NaT
data_2019['Billing_Time'] = np.NaN #pd.NaT

# create column to store cumulative
# minute duration of patient processes
data_2019['cumulative_minutes'] = pd.NaT

# create column to timestamp end time of patient
# journey--look into best way to structure this
data_2019['Cumsum_Time_Timestamp'] = pd.NaT

# create second datetime column, store as date, so we can
# use the duplicate column as we wish for additional operations
data_2019["date"] = data_2019["datetime"]


# setting global parameters for number of hospital staff
CHECKIN_CAPACITY = 2
NURSE_CAPACITY = 12
DOCTOR_CAPACITY = 6
TECHNICIAN_CAPACITY = 2
BILLING_CAPACITY = 6
SIMULATION_TIME = 518400 # one year expressed in minutes
start_of_year = datetime(2019, 1, 1, 0, 0, 0) #pd.Timestamp(2019, 1, 1, 0, 0, 0)\

# Adding the 'cumulative_minutes' column to show minute count for arrival
# NEED TO RENAME THIS PART
# Move to position for arrival time, this is what it really represents anyway
data_2019['cumulative_minutes'] = (data_2019['datetime'] - start_of_year).dt.total_seconds() / 60

# stripping date for preservation needs to be
# moved to new position, perhaps col 1
# rename date_arrive
data_2019['date'] = data_2019['date'].dt.date

# make col 2, rename hour_arrive
# transform datetime column to datetime dtype to show time in only H/M/S
# work with timedelta as we proceed to record time durations of processes
data_2019['datetime'] = pd.to_datetime(data_2019['datetime']).dt.floor('min')
data_2019['datetime'] = data_2019['datetime'].apply(lambda dt: pd.Timestamp(dt).time())


# current v1
def run_simulation(data_2019):
    # Initialize simpy environment
    env = simpy.Environment()

    # Define resources
    checkin_resource = simpy.Resource(env, capacity=CHECKIN_CAPACITY)
    nurse_resource = simpy.Resource(env, capacity=NURSE_CAPACITY)
    technician_resource = simpy.Resource(env, capacity=TECHNICIAN_CAPACITY)
    doctor_resource = simpy.Resource(env, capacity=DOCTOR_CAPACITY)
    billing_resource = simpy.Resource(env, capacity=BILLING_CAPACITY)
    # random_wait_store = simpy.Store(env, capacity=75)

    # Loop through each patient in the data and process them using the appropriate generator
    for index, row in data_2019.iterrows():
        patient_type = row['patient_type']

        # Construct a datetime object to show cumulative minute
        # marks patient arrival based on 01-01-19 00:00:00 starttime
        # intended_arrival_time = row['cumulative_minutes'] / 60
        # intended_arrival_time = row['cumulative_minutes'] * 60
        if patient_type == 'A':
            env.process(
                patient_type_A(env, index, checkin_resource, nurse_resource, billing_resource, data_2019, index))
        elif patient_type == 'B':
            env.process(patient_type_B(env, index, checkin_resource, nurse_resource, doctor_resource, billing_resource,
                                       data_2019, index))
        elif patient_type == 'C':
            env.process(
                patient_type_C(env, index, checkin_resource, nurse_resource, doctor_resource, technician_resource,
                               billing_resource, data_2019, index))
        elif patient_type == 'D':
            env.process(
                patient_type_D(env, index, technician_resource, nurse_resource, doctor_resource, billing_resource,
                               data_2019, index))

    # Run the simulation
    env.run(until=SIMULATION_TIME)
    save_to_json_file()

    # Initialize a JSON flow file, review outputs.json to debug process issues
json_flow = {
    "events": []
}

# define logger for JSON
json_flow = {
    "events": []
}

# define logger for JSON

def log_event(event_description):
    """Utility function to log events to the global json_flow."""
    global json_flow
    json_flow["events"].append(event_description)

def Random_Wait(env, patient_id):
    yield env.timeout(random.expovariate(1/50)) 
    
def Checkin(env, checkin_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Checkin", "time": env.now})
    checkin_request = checkin_resource.request()
    yield checkin_request
    log_event({"patient_id": patient_id, "action": "starting Checkin", "time": env.now})
    yield env.timeout(random.randint(6, 18))
    checkin_resource.release(checkin_request)
    log_event({"patient_id": patient_id, "action": "finished Checkin", "time": env.now})

def Billing(env, billing_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Billing", "time": env.now})
    billing_request = billing_resource.request()
    yield billing_request
    log_event({"patient_id": patient_id, "action": "starting Billing", "time": env.now})
    yield env.timeout(random.randint(3, 12))
    billing_resource.release(billing_request)
    log_event({"patient_id": patient_id, "action": "finished Billing", "time": env.now})

def Nurse_Consult(env, nurse_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Nurse_Consult", "time": env.now})
    nurse_request = nurse_resource.request()
    yield nurse_request
    log_event({"patient_id": patient_id, "action": "starting Nurse_Consult", "time": env.now})
    yield env.timeout(random.expovariate(1/25))
    nurse_resource.release(nurse_request)
    log_event({"patient_id": patient_id, "action": "finished Nurse_Consult", "time": env.now})

def Nurse_Injection(env, nurse_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Nurse_Injection", "time": env.now})
    nurse_request = nurse_resource.request()
    yield nurse_request
    log_event({"patient_id": patient_id, "action": "starting Nurse_Injection", "time": env.now})
    yield env.timeout(random.randint(3, 8))
    nurse_resource.release(nurse_request)
    log_event({"patient_id": patient_id, "action": "finished Nurse_Injection", "time": env.now})
    
def Nurse_Surgery(env, nurse_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Nurse_Surgery", "time": env.now})
    nurse_request = nurse_resource.request()
    yield nurse_request
    log_event({"patient_id": patient_id, "action": "starting Nurse_Surgery", "time": env.now})
    yield env.timeout(random.expovariate(1/80))
    nurse_resource.release(nurse_request)
    log_event({"patient_id": patient_id, "action": "finished Nurse_Surgery", "time": env.now})

def Doctor_Surgery(env, doctor_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Doctor_Surgery", "time": env.now})
    doctor_request = doctor_resource.request()
    yield doctor_request
    log_event({"patient_id": patient_id, "action": "starting Doctor_Surgery", "time": env.now})
    yield env.timeout(random.expovariate(1/80))
    doctor_resource.release(doctor_request)
    log_event({"patient_id": patient_id, "action": "finished Doctor_Surgery", "time": env.now})

def Doctor_Consult(env, doctor_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Doctor_Consult", "time": env.now})
    doctor_request = doctor_resource.request()
    yield doctor_request
    log_event({"patient_id": patient_id, "action": "starting Doctor_Consult", "time": env.now})
    yield env.timeout(random.randint(6, 12))
    doctor_resource.release(doctor_request)
    log_event({"patient_id": patient_id, "action": "finished Doctor_Consult", "time": env.now})

def Checkout_Consult(env, nurse_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Checkout_Consult", "time": env.now})
    checkout_request = nurse_resource.request()
    yield checkout_request
    log_event({"patient_id": patient_id, "action": "starting Checkout_Consult", "time": env.now})
    yield env.timeout(random.randint(6, 18))
    nurse_resource.release(checkout_request)
    log_event({"patient_id": patient_id, "action": "finished Checkout_Consult", "time": env.now})

def Surgical_Dressing(env, doctor_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Surgical_Dressing", "time": env.now})
    dressing_request = doctor_resource.request()
    yield dressing_request
    log_event({"patient_id": patient_id, "action": "starting Surgical_Dressing", "time": env.now})
    yield env.timeout(random.randint(6, 18))
    doctor_resource.release(dressing_request)
    log_event({"patient_id": patient_id, "action": "finished Surgical_Dressing", "time": env.now})

def Surgery_Recovery(env, nurse_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Surgery_Recovery", "time": env.now})
    recovery_request = nurse_resource.request()
    yield recovery_request
    log_event({"patient_id": patient_id, "action": "starting Surgery_Recovery", "time": env.now})
    yield env.timeout(random.randint(60, 180))
    nurse_resource.release(recovery_request)
    log_event({"patient_id": patient_id, "action": "finished Surgery_Recovery", "time": env.now})

def Technician_Scan_Consult(env, technician_resource, patient_id):
    log_event({"patient_id": patient_id, "action": "requesting Technician_Scan_Consult", "time": env.now})
    technician_request = technician_resource.request()
    yield technician_request
    log_event({"patient_id": patient_id, "action": "starting Technician_Scan_Consult", "time": env.now})
    #yield env.timeout(random.expovariate(1/25))
    yield env.timeout(random.randint(20, 88))
    technician_resource.release(technician_request)
    log_event({"patient_id": patient_id, "action": "finished Technician_Scan_Consult", "time": env.now})
    
json_flow = {
    "events": []
}  
    
#def save_to_json_file():
#    with open('output.json', 'w') as file:
#        json.dump(json_flow, file, indent=4)
        
def save_to_json_file():
    # Save json_flow to a gzip compressed JSON file
    with gzip.open('output.json.gz', 'wb') as file:
        file.write(json.dumps(json_flow, indent=4).encode('utf-8'))

#current
ER_processes = {
    "Random_Wait": lambda env, patient_id: Random_Wait(env, patient_id),
    "Checkin": lambda env, checkin_resource, patient_id: env.process(Checkin(env, checkin_resource, patient_id)),
    "Nurse_Consult": lambda env, nurse_resource, patient_id: env.process(Nurse_Consult(env, nurse_resource, patient_id)),
    "Nurse_Injection": lambda env, nurse_resource, patient_id: env.process(Nurse_Injection(env, nurse_resource, patient_id)),
    "Technician_Scan_Consult": lambda env, technician_resource, patient_id: env.process(Technician_Scan_Consult(env, technician_resource, patient_id)),
    "Doctor_Consult": lambda env, doctor_resource, patient_id: env.process(Doctor_Consult(env, doctor_resource, patient_id)),
    "Nurse_Surgery": lambda env, nurse_resource, patient_id: env.process(Nurse_Surgery(env, nurse_resource, patient_id)),
    "Doctor_Surgery": lambda env, doctor_resource, patient_id: env.process(Doctor_Surgery(env, doctor_resource, patient_id)),
    "Surgical_Dressing": lambda env, doctor_resource, patient_id: env.process(Surgical_Dressing(env, doctor_resource, patient_id)),
    "Surgery_Recovery": lambda env, nurse_resource, patient_id: env.process(Surgery_Recovery(env, nurse_resource, patient_id)),
    "Checkout_Consult": lambda env, nurse_resource, patient_id: env.process(Checkout_Consult(env, nurse_resource, patient_id)),
    "Billing": lambda env, billing_resource, patient_id: env.process(Billing(env, billing_resource, patient_id))
}


# current

def patient_type_A(env, patient_id, checkin_resource, nurse_resource, billing_resource, data_2019, index):
    wait_time_event = ER_processes["Random_Wait"](env, index)  # (env, patient_id, random_wait_store)
    yield env.timeout(data_2019.at[index, 'cumulative_minutes'] - env.now)
    yield from wait_time_event
    data_2019.at[index, 'Wait_Time'] = env.now
    log_event({"patient_id": index, "action": "Random_Wait", "time": env.now})

    yield ER_processes["Checkin"](env, checkin_resource, index)
    data_2019.at[index, 'Checkin_Time'] = env.now  # /(10**4)
    log_event({"patient_id": index, "action": "Checkin", "time": env.now})

    yield ER_processes["Nurse_Injection"](env, nurse_resource, index)
    data_2019.at[index, 'Nurse_Injection_Time'] = env.now
    log_event({"patient_id": index, "action": "Nurse_Injection", "time": env.now})

    yield ER_processes["Checkout_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Checkout_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkout_Consult", "time": env.now})

    yield ER_processes["Billing"](env, billing_resource, index)
    data_2019.at[index, 'Billing_Time'] = env.now
    log_event({"patient_id": index, "action": "Billing", "time": env.now})


def patient_type_B(env, patient_id, checkin_resource, nurse_resource, doctor_resource, billing_resource, data_2019,
                   index):
    wait_time_event = ER_processes["Random_Wait"](env, index)
    yield env.timeout(data_2019.at[index, 'cumulative_minutes'] - env.now)
    yield from wait_time_event
    data_2019.at[index, 'Wait_Time'] = env.now
    log_event({"patient_id": index, "action": "Random_Wait", "time": env.now})

    yield ER_processes["Checkin"](env, checkin_resource, index)
    data_2019.at[index, 'Checkin_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkin", "time": env.now})

    yield ER_processes["Nurse_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Nurse_Consult_Time'] = env.now
    log_event({"patient_id": index, "action": "Nurse_Consult", "time": env.now})

    yield ER_processes["Doctor_Consult"](env, doctor_resource, index)
    data_2019.at[index, 'Doctor_Consult_Time'] = env.now
    log_event({"patient_id": index, "action": "Doctor_Consult", "time": env.now})

    yield ER_processes["Checkout_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Checkout_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkout_Consult", "time": env.now})

    yield ER_processes["Billing"](env, billing_resource, index)
    data_2019.at[index, 'Billing_Time'] = env.now
    log_event({"patient_id": index, "action": "Billing", "time": env.now})


def patient_type_C(env, patient_id, checkin_resource, nurse_resource, doctor_resource, technician_resource,
                   billing_resource, data_2019, index):
    wait_time_event = ER_processes["Random_Wait"](env, index)
    yield env.timeout(data_2019.at[index, 'cumulative_minutes'] - env.now)
    yield from wait_time_event
    data_2019.at[index, 'Wait_Time'] = env.now
    log_event({"patient_id": index, "action": "Random_Wait", "time": env.now})

    yield ER_processes["Checkin"](env, checkin_resource, index)
    data_2019.at[index, 'Checkin_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkin", "time": env.now})

    yield ER_processes["Nurse_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Nurse_Consult_Time'] = env.now
    log_event({"patient_id": index, "action": "Nurse_Consult", "time": env.now})

    yield ER_processes["Doctor_Consult"](env, doctor_resource, index)
    data_2019.at[index, 'Doctor_Consult_Time'] = env.now
    log_event({"patient_id": index, "action": "Doctor_Consult", "time": env.now})

    yield ER_processes["Technician_Scan_Consult"](env, technician_resource, index)
    data_2019.at[index, 'Technician_Scan_Time'] = env.now
    log_event({"patient_id": index, "action": "Technician_Scan_Consult", "time": env.now})

    yield ER_processes["Checkout_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Checkout_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkout_Consult", "time": env.now})

    yield ER_processes["Billing"](env, billing_resource, index)
    data_2019.at[index, 'Billing_Time'] = env.now
    log_event({"patient_id": index, "action": "Billing", "time": env.now})


def patient_type_D(env, patient_id, technician_resource, nurse_resource, doctor_resource, billing_resource, data_2019,
                   index):
    wait_time_event = ER_processes["Random_Wait"](env, index)
    yield env.timeout(data_2019.at[index, 'cumulative_minutes'] - env.now)
    yield from wait_time_event
    data_2019.at[index, 'Wait_Time'] = env.now
    log_event({"patient_id": index, "action": "Random_Wait", "time": env.now})

    yield ER_processes["Technician_Scan_Consult"](env, technician_resource, index)
    data_2019.at[index, 'Technician_Scan_Time'] = env.now
    log_event({"patient_id": index, "action": "Technician_Scan_Consult", "time": env.now})

    yield ER_processes["Nurse_Surgery"](env, nurse_resource, index)
    data_2019.at[index, 'Nurse_Surgery_Time'] = env.now
    log_event({"patient_id": index, "action": "Nurse_Surgery", "time": env.now})

    yield ER_processes["Doctor_Surgery"](env, doctor_resource, index)
    data_2019.at[index, 'Doctor_Surgery_Time'] = env.now
    log_event({"patient_id": index, "action": "Doctor_Surgery", "time": env.now})

    yield ER_processes["Surgical_Dressing"](env, doctor_resource, index)
    data_2019.at[index, 'Surgery_Dress_Time'] = env.now
    log_event({"patient_id": index, "action": "Surgical_Dressing", "time": env.now})

    yield ER_processes["Surgery_Recovery"](env, nurse_resource, index)
    data_2019.at[index, 'Surgery_Recover_Time'] = env.now
    log_event({"patient_id": index, "action": "Surgery_Recovery", "time": env.now})

    yield ER_processes["Checkout_Consult"](env, nurse_resource, index)
    data_2019.at[index, 'Checkout_Time'] = env.now
    log_event({"patient_id": index, "action": "Checkout_Consult", "time": env.now})

    yield ER_processes["Billing"](env, billing_resource, index)
    data_2019.at[index, 'Billing_Time'] = env.now
    log_event({"patient_id": index, "action": "Billing", "time": env.now})

run_simulation(data_2019)

# transform Arrival Time to timedelta for operation with cumsum
data_2019['Arrival_Time'] = data_2019['cumulative_minutes'] #.apply(lambda x: pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second))

data_2019['Cumsum_Time_Timestamp'] = pd.to_datetime(data_2019['datetime'].astype(str)) + pd.to_timedelta(data_2019['Billing_Time'] - data_2019['Arrival_Time'], unit='m')

data_2019['visit_dur'] = data_2019['Cumsum_Time_Timestamp'] - pd.to_timedelta(data_2019['Arrival_Time'], unit='m')

# Calculate the difference in time
data_2019['visit_dur'] = data_2019['Cumsum_Time_Timestamp'] - pd.to_datetime(data_2019['datetime'].astype(str))

# Convert the timedelta to minutes and round it
data_2019['visit_dur_total'] = data_2019['visit_dur'].dt.total_seconds() / 60
data_2019['visit_dur_total'] = data_2019['visit_dur_total'].round()