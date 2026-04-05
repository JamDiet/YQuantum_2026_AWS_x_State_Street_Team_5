import pandas as pd

df3 = pd.read_csv('quantum_outputs/part1_spec_test.csv')
df4 = pd.read_csv('quantum_outputs/part1_spec_train.csv')
df5 = pd.read_csv('quantum_outputs/part2_mru_features.csv')
df6 = pd.read_csv('quantum_outputs/part2_mru_predictions.csv')

print('spec_test')
print(df3.head, df3.columns)
print('spec_train')
print(df4.head, df4.columns)
print('mru_features')
print(df5.head, df5.columns)
print('mru_predictions')
print(df6.head, df6.columns)