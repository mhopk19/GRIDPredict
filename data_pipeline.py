import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataPipeline:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, low_memory=False)
        print(f'{filepath} : {self.df.shape}')
        
    def rename_columns(self):
        self.df.columns = [i.replace(' [kW]', '') for i in self.df.columns]
        
    def aggregate_columns(self):
        self.df['Furnace'] = self.df[['Furnace 1', 'Furnace 2']].sum(axis=1)
        self.df['Kitchen'] = self.df[['Kitchen 12', 'Kitchen 14', 'Kitchen 38']].sum(axis=1)
        self.df.drop(['Furnace 1', 'Furnace 2', 'Kitchen 12', 'Kitchen 14', 'Kitchen 38', 'icon', 'summary'], axis=1, inplace=True)
        
    def handle_missing_data(self):
        self.df = self.df[0:-1]
        self.df['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
        self.df['cloudCover'] = self.df['cloudCover'].astype('float')
        
    def transform_time(self):
        self.df['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(self.df),  freq='min'))
        self.df['year'] = self.df['time'].apply(lambda x: x.year)
        self.df['month'] = self.df['time'].apply(lambda x: x.month)
        self.df['day'] = self.df['time'].apply(lambda x: x.day)
        self.df['weekday'] = self.df['time'].apply(lambda x: x.day_name())
        self.df['weekofyear'] = self.df['time'].apply(lambda x: x.weekofyear)
        self.df['hour'] = self.df['time'].apply(lambda x: x.hour)
        self.df['minute'] = self.df['time'].apply(lambda x: x.minute)
        
    def hours2timing(self, x):
        if x in [22, 23, 0, 1, 2, 3]:
            return 'Night'
        elif x in range(4, 12):
            return 'Morning'
        elif x in range(12, 17):
            return 'Afternoon'
        elif x in range(17, 22):
            return 'Evening'
        else:
            return 'X'

    def assign_timing(self):
        self.df['timing'] = self.df['hour'].apply(self.hours2timing)
        
    def plot_corr(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr[corr > 0.9], vmax=1, vmin=-1, center=0, ax=ax)
        plt.show()

    def final_transformations(self):
        self.df['use_HO'] = self.df['use']
        self.df['gen_Sol'] = self.df['gen']
        self.df.drop(['use', 'House overall', 'gen', 'Solar'], axis=1, inplace=True)
        
    def save_to_csv(self, filepath="cleaned_HomeC.csv"):
        self.df.to_csv(filepath, index=False)
        
    def run_pipeline(self):
        self.rename_columns()
        self.aggregate_columns()
        self.handle_missing_data()
        self.transform_time()
        self.assign_timing()
        self.plot_corr()
        self.final_transformations()
        self.save_to_csv()

# Usage
pipeline = DataPipeline("input/HomeC.csv")
pipeline.run_pipeline()

