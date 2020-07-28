import pandas as pd

path = 'variance.csv'
tpath = 'output.csv'
#tpath = 'temp.csv'

var_df = pd.read_csv(path)
temp_df = pd.read_csv(tpath)

ans_df = pd.merge(var_df, temp_df, on='Dataset', how='left')

print(ans_df)
ans_df.to_csv('ans.csv', index=False)
#columns = var_df.columns

#print('columns = ' + str(columns))
#
#print(var_df)
#print(temp_df)
#
#data = []
#
#for index,row in var_df.iterrows():
#    print('row = ')
#    print(row)
#    print('index = ' + str(index))
#    dataset = row['dataset/d2h']
#    days = [1,7,14,30, 90, 180, 365]
#    for day in days:
#        if (dataset.startswith(str(day) +  ' days')):
#            label = dataset[(len(str(day)) + 6):]
#            print('label = ' + str(label))
#            temp_label = str(day) + '_' + label
#            print('temp_label = ' + str(temp_label))
#            temp_row = temp_df.loc[temp_df['project'] == temp_label]
#            print('temp_row')
#            print(temp_row)
#            print('temp_dim ')
#            print(temp_row['original dim'].item())
#            row['original_dim'] = temp_row['original dim'].item()
#            row['intrinsic dim_L1'] = temp_row['intrinsic dim_L1'].item()
#            row['intrinsic dim_L2'] = temp_row['intrinsic dim_L2'].item()
#    rowa = row.tolist()
#    print('rowa = ')
#    print(rowa)
#    data.append(rowa)
#
#output_df = pd.DataFrame(data, columns=columns)
#print(output_df)
#output_df.to_csv('output.csv', index=False)
