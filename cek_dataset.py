import pandas as pd

# baca dataset
df = pd.read_csv("train.csv")

# tampilkan 5 data pertama
print(df.head())

# tampilkan nama kolom
print(df.columns)

# tampilkan jumlah data
print(df.shape)