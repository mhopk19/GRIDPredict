import pandas as pd
class PowerDataFrame():
    def __init__(self, data_file):
          self.data = pd.read_csv(data_file, low_memory = False)
          self.rows = len(self.data)
          self.columns = self.data.columns

    
    def get_interval_data(self, hourly = True):
        """
        hourly_constant: whether or not the function returns data by the hour or every 15 minutes
        """
        if (hourly):
             interval_constant = 60
        else:
             interval_constant = 15
        rows_per_val = 60 * interval_constant # 60 seconds x 60/15 minutes

        i = 0
        interval_df = pd.DataFrame()
        while(i < len(self.data)):
            upper_max = min(i+rows_per_val, len(self.data)-1)
            new_comp = self.data.iloc[i:upper_max].sum(axis=0)
            frames = [interval_df, new_comp]
            interval_df = pd.concat(frames)
            i = i + rows_per_val
        
        return interval_df