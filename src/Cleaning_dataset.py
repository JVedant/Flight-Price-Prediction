import pandas as pd
import numpy as np

import config

if __name__ == "__main__":

   # reading the data
   dataset = pd.read_csv(config.DATASET_PATH)

   # deleting the null rows
   dataset.dropna(inplace=True)

   # Converting Journey date and time data to numerical
   dataset["Journey_day"] = pd.to_datetime(dataset.Date_of_Journey, format="%d/%m/%Y").dt.day
   dataset["Journey_month"] = pd.to_datetime(dataset["Date_of_Journey"], format = "%d/%m/%Y").dt.month
   dataset.drop('Date_of_Journey', 1, inplace=True)

   # Converting departure time  to numerical
   dataset["Dep_hour"] = pd.to_datetime(dataset["Dep_Time"]).dt.hour
   dataset["Dep_min"] = pd.to_datetime(dataset["Dep_Time"]).dt.minute
   dataset.drop(["Dep_Time"], axis = 1, inplace = True)

   # Converting Arrival time  to numerical
   dataset["Arr_hour"] = pd.to_datetime(dataset["Arrival_Time"]).dt.hour
   dataset["Arr_min"] = pd.to_datetime(dataset["Arrival_Time"]).dt.minute
   dataset.drop(["Arrival_Time"], axis = 1, inplace = True)

   # cleaning the duration and converting to numerical
   duration = list(dataset['Duration'])
   for i in range(len(duration)):
      if len(duration[i].split()) != 2:
         if "h" in duration[i]:
               duration[i] = duration[i].strip() + " 0m"
         else:
               duration[i] = "0h " + duration[i]
   duration_hours = []
   duration_mins = []
   for i in range(len(duration)):
      duration_hours.append(int(duration[i].split(sep = "h")[0]))
      duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))

   dataset["Duration_hrs"] = duration_hours
   dataset["Duration_mins"] = duration_mins
   dataset.drop('Duration', 1, inplace=True)


   # concatting the encoded data
   dataset = pd.concat([dataset, pd.get_dummies(dataset[['Airline', 'Source', 'Destination']],
        drop_first = True
    )], axis=1)
   dataset.drop(['Airline', 'Source', 'Destination'], 1, inplace=True)

   # Dropping the features which are not important
   dataset.drop(['Route', 'Additional_Info'], 1, inplace=True)


   # mapping for the total_stops
   stops_mapping = {
      'non-stop': 0,
      '1 stop': 1,
      '2 stops': 2,
      '3 stops': 3,
      '4 stops': 4,
   }
   # maping it to dataset
   dataset['Stops'] = dataset.Total_Stops.map(stops_mapping)
   dataset.drop('Total_Stops', 1, inplace=True)


   # save the dataset for training
   dataset.to_csv(config.TRAINING_PATH, index=False)