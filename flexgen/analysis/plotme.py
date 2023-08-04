import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('output.csv')
df['TIME'] = pd.to_datetime(df['TIME'])
plt.figure(figsize=(10, 6))
plt.plot(df['TIME'], df['i'], label='i')
plt.plot(df['TIME'], df['j'], label='j')
plt.plot(df['TIME'], df['k'], label='k')
plt.xlabel('TIME')
plt.ylabel('i, j, k Values')
plt.title('i, j, k Values Over Time')
plt.legend()
plt.tight_layout()
plt.show()
