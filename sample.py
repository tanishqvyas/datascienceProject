import numpy as np
import pandas as pd
import ezodf

doc = ezodf.opendoc(r'D:\Projects\Data_Science_Assignment\student_por.ods')

sheet = doc.sheets[0]
df_dict = {}
for i, row in enumerate(sheet.rows()):
    # row is a list of cells
    # assume the header is on the first row
    if i == 0:
        # columns as lists in a dictionary
        df_dict = {cell.value:[] for cell in row}
        # create index for the column headers
        col_index = {j:cell.value for j, cell in enumerate(row)}
        continue
    for j, cell in enumerate(row):
        # use header instead of column index
        df_dict[col_index[j]].append(cell.value)
# and convert to a DataFrame

df = pd.DataFrame(df_dict)

for col in {'Pstatus','Medu','Fedu','Mjob','Fjob','reason','paid','romantic','famrel','Dalc'}:
    ori_rat = df[col].isna().mean()

    if ori_rat >= 0.03: continue

    add_miss_rat = (0.03 - ori_rat) / (1 - ori_rat)
    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
    df.loc[vals_to_nan, col] = np.NaN

for col in {'famsize','traveltime','studytime','goout','health','famsup'}:
    ori_rat = df[col].isna().mean()

    if ori_rat >= 0.015: continue

    add_miss_rat = (0.015 - ori_rat) / (1 - ori_rat)
    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
    df.loc[vals_to_nan, col] = np.NaN

for col in {'activities','internet','freetime','guardian'}:
    ori_rat = df[col].isna().mean()

    if ori_rat >= 0.010: continue

    add_miss_rat = (0.010 - ori_rat) / (1 - ori_rat)
    vals_to_nan = df[col].dropna().sample(frac=add_miss_rat).index
    df.loc[vals_to_nan, col] = np.NaN

print(df)

df.fillna("NaN", inplace = True)

df.to_csv(r'D:\Projects\Data_Science_Assignment\datascienceProject\student_data.csv')
