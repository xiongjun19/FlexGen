import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('output.csv')
df1['TIME'] = pd.to_datetime(df1['TIME'])
df2 = pd.read_csv('mem-run.csv')
df2['TIME'] = pd.to_datetime(df2['TIME'])
merged_df = pd.merge_asof(df2, df1, on='TIME', direction='nearest')

plt.figure(figsize=(10, 6))
plt.plot(df1['TIME'], df1['j'], label='j: j-th layer (total:130)')
plt.plot(df1['TIME'], df1['k'], label='k: k-th gpu batch (total: 4)')

plt.plot(merged_df['TIME'], merged_df['i'], label='i: the i-th token (total 8)')
plt.plot(merged_df['TIME'], 2*merged_df['PCI_TX_MBps']/1024, label='PCI_TX_2xGBps')
plt.plot(merged_df['TIME'], 2*merged_df['PCI_RX_MBps']/1024, label='PCI_RX_2xGBps')
plt.plot(merged_df['TIME'], merged_df['MEM%'], label='MEM%')
plt.plot(merged_df['TIME'], merged_df['GPU%'], label='GPU%')
plt.plot(merged_df['TIME'], merged_df['GPUMEM%'], label='GPUMEM%')
plt.plot(merged_df['TIME'], merged_df['CPU%'], label='CPU%')

plt.xlabel('TIME')
plt.ylabel('Values')
plt.title('Details of MEM (NORMAL MEMORY) OPT-66B INFERENCE Values Over Time')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
