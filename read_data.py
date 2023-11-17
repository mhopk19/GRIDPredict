import pandas as pd
import classes.powerdf as pdf

# example of using power data frame

power_df = pdf.PowerDataFrame("HomeC.csv")
# hourly=True returns dataframe with hourly energy values, hourly=False returns dataframe with 15 minute interval values
df = power_df.get_interval_data(hourly = True)
print("df rows", len(df))
    