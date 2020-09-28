import pandas as pd

dates=(
    "2020-04-01", "2020-04-08" ,"2020-04-15" ,"2020-04-22", "2020-04-29",
    "2020-05-06" ,"2020-05-13" ,"2020-05-20","2020-05-27","2020-06-03",
    "2020-06-10", "2020-06-17", "2020-06-24", "2020-07-01", "2020-07-08",
    "2020-07-15", "2020-07-22")
#days=(
#   60, 67, 74, 81, 88, 95, 102, 109, 116, 123, 130, 
#    137, 144, 151, 158, 165, 172)

df = pd.DataFrame()
for date in dates:
    df_file = pd.read_csv("./analysis/UoA_"+date+"local_obs.csv", index_col=0)
    df_file['model.id'] = "UoA"
    df_file['data date'] = date
    df = df.append(df_file)

df['data date'] = pd.to_datetime(df['data date'])
df['onset date'] = pd.to_datetime(df['onset date'])
mask = df['onset date'].values >= (df['data date'].values - pd.to_timedelta(7, unit='days'))
df = df.loc[mask]


df['data date'] = [val.strftime("%Y-%m-%d") for val in df['data date']]
df['onset date'] = [val.strftime("%Y-%m-%d") for val in df['onset date']]

print(df)
df.to_csv("./analysis/cprs/UoA_2020-09-28.csv")