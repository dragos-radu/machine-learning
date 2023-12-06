import pandas as pd

data = {'Nume': ['Ana', 'Bogdan', 'Cristina', 'Ana', 'Cristina', None],
        'Varsta': [25, 30, 22, 25, 30, None],
        'Salariu': [50000, 60000, 45000, 50000, 60000, None]}

df = pd.DataFrame(data)
df2 = df.drop_duplicates()
#print(df2)

#print(df.dropna())
#print(df.fillna(0))

#print(df.rename(columns={'Nume': 'Numele'})) #cu inplace true se modifica si originalul



df3 = pd.DataFrame(data)
#print(df3)
#dep = df3.groupby('Dep')

#for nume, grup in dep:
    #print(grup)


#medie_sal = dep['Salariu'].mean()
#(medie_sal)

#agg = dep.agg({'Varsta': 'mean', 'Salariu': ['sum', 'median'], 'Nume':'count'})
#print(agg)

data = {
    'Data': ['2022-01-01', '2022-02-01', '2022-03-01'],
    'Vanzari': [100,150,200]
}
df4 = pd.DataFrame(data)

df4['Data'] = pd.to_datetime(df4['Data'])
df4['Zi'] = df4['Data'].dt.day
df4['Luna'] = df4['Data'].dt.month
df4['An'] = df4['Data'].dt.year


df4['dif-zi'] = (df4['Data'] - pd.to_datetime('2022-01-01')).dt.days
#print(df4)



df5 = pd.DataFrame({
    'Nume': ['Ana', 'Bogdan'],
    'Varsta': [35, 40],
    'ID': [1,2]
})

df6 = pd.DataFrame({
    'Nume': ['Cristina', 'David'],
    'Varsta': [38, 30],
    'ID': [2,3]
})

df_c = pd.merge(df5, df6, on ='ID', how='inner')
#print(df_c)

data = {'Nume': ['Ana', 'Bogdan', 'David', 'Elena', 'Florin'],
        'Varsta': [25, 30, 22, 25, 30],
        'Salariu': [50000, 60000, 45000, 50000, 60000],
        'Dep': ['IT', 'HR', 'IT', 'IT', 'HR']}

df.to_csv('date.csv', index=False)




