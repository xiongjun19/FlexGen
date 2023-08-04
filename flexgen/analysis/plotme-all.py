import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('output.csv')
df1['TIME'] = pd.to_datetime(df1['TIME'])

df2 = pd.read_csv('mem-run.csv')
df2['TIME'] = pd.to_datetime(df2['TIME'])


plt.figure(figsize=(10, 6))
plt.plot(df1['TIME'], df1['i'], label='i: the i-th token')
plt.plot(df1['TIME'], df1['j'], label='j: j-th layer')
plt.plot(df1['TIME'], df1['k'], label='k: k-th gpu batch')
plt.plot(df2['TIME'], df2['MEM%'], label='MEM%')

plt.xlabel('TIME')
plt.ylabel('Values')
plt.title('i, j, k, and MEM% Values Over Time')
plt.legend()
plt.tight_layout()
plt.show()
