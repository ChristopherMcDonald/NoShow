import sys
sys.path.append('./../Data/')
# Importing Standard ML Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

np.random.seed(55) # for consistency in training

# getData - reads, processes and cleans the data for a variety of ML use cases
def getData(balance):
    # Importing the dataset
    df = pd.read_csv('./../Data/data.csv')

    # Balancing the dataset due to large portion of people showing up
    if balance:
        df = df.drop(df[df["No-show"] == 'No'].sample(frac = 0.75).index)

    # Converts Strings to Date objects
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

    # What day was the appointment booked for? (e.g. 1, 23, 30, ...)
    df['AppointmentDay-DayOfMonth'] = df.apply(lambda row: row['AppointmentDay'].day, axis = 1) # SO to Eric

    # What day was the aappointment booked on? (e.g. 1, 23, 30, ...)
    df['ScheduledDay-DayOfMonth'] = df.apply(lambda row: row['ScheduledDay'].day, axis = 1) # SO to Eric

    # How many seconds between the appointment and book date?
    df['TimeDifference'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.seconds # SO to Eric

    # Maps M and F to a 1 or a 0
    gender = preprocessing.LabelEncoder()
    gender.fit(['M', 'F'])
    df.Gender = gender.transform(df.Gender)

    # Maps Yes and NO to a 1 or 0
    noshow = preprocessing.LabelEncoder()
    noshow.fit(['Yes', 'No'])
    df['No-show'] = noshow.transform(df['No-show'])

    # One-Hot encoding both types of dates on which weekday they fall on and month they are on in
    # This will generate columns for each day and month for both types of dates, which contains a 1 based on whether
    # the appointment/schedule fell on that day or month
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    scheduleDayWeekDay = {}
    for i in range(0, 7):
        df['ScheduledDay-' + days[i]] = 0
        scheduleDayWeekDay[i] = 'ScheduledDay-' + days[i]

    scheduleDayMonth = {}
    for i in range(1, 13):
        df['ScheduledDay-' + months[i - 1]] = 0
        scheduleDayMonth[i] = 'ScheduledDay-' + months[i - 1]

    appointDayWeekDay = {}
    for i in range(0, 7):
        df['AppointmentDay-' + days[i]] = 0
        appointDayWeekDay[i] = 'AppointmentDay-' + days[i]

    appointDayMonth = {}
    for i in range(1, 13):
        df['AppointmentDay-' + months[i - 1]] = 0
        appointDayMonth[i] = 'AppointmentDay-' + months[i - 1]

    for i, row in df.iterrows():
        # note use of _ throwaway variable to suppress output from df.set_value
        _ = df.set_value(i, scheduleDayWeekDay[row['ScheduledDay'].weekday()], 1)
        _ = df.set_value(i, scheduleDayMonth[row['ScheduledDay'].month], 1)
        _ = df.set_value(i, appointDayWeekDay[row['AppointmentDay'].weekday()], 1)
        _ = df.set_value(i, appointDayMonth[row['AppointmentDay'].month], 1)

    # removing appointmentID and patientID due to uniqueness, TODO have a column for having missed an appointment before
    # removing schedule and appointment dates as these are too involved, seperated into parts already
    # removing target variable
    X = [x for x in df.columns if (x != 'AppointmentID' and x != 'PatientId' and x != 'Neighbourhood' and x != 'ScheduledDay' and x != 'AppointmentDay' and x != 'No-show')]
    X = df[X]

    # isolate target variable
    Y = df.iloc[:, 13]

    return X, Y
