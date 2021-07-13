import pandas as pd


df = pd.read_csv('temp.csv')
df['count'] = 1
df['percent'] = ((1 / df.shape[0]) * 100)
print(df)
df_gr = df.groupby(['real', 'predict']).sum()
df_gr['percent'] = df_gr['percent'].astype(int)
print(df_gr)