import csv
import numpy as np
import pandas as pd
#from uszipcode import SearchEngine


# get LAT LONG from zipcodes:
# zipcode = ZCDB[54115]
# zipcode.latitude, zipcode.longitude

#ZCDB = ZipCodeDatabase()


data = np.matrix([1, 2, 3, 4])

with open('../data/predictions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            new = np.array([row[0], row[-3], row[-2], row[-1]])
            data = np.vstack((data, new))
            line_count += 1
    print(f'Processed {line_count} lines.')

data = np.delete(data, (0), axis=0)
dataset = pd.DataFrame(data, columns=['name', 'zip', 'p', 'prediction'])
#search = SearchEngine()


import plotly
import plotly.figure_factory as ff

df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/laucnty16.csv')
df_sample['State FIPS Code'] = df_sample['State FIPS Code'].apply(lambda x: str(x).zfill(2))
df_sample['County FIPS Code'] = df_sample['County FIPS Code'].apply(lambda x: str(x).zfill(3))
df_sample['FIPS'] = df_sample['State FIPS Code'] + df_sample['County FIPS Code']

out = [0]*len(df_sample['Unemployment Rate (%)'])

zip2fip = pd.read_csv("../data/zip_fips.csv")


fips = np.array(df_sample['FIPS'].astype(str).tolist())

for row in dataset.itertuples():
    S = zip2fip.loc[zip2fip['zip'] == int(row.zip), 'county']
    if not(S.empty):
        L = list(np.where(fips == str(S.iloc[0])))
        if L[0] != None and row.prediction == 'True':
            out[int(L[0])] += 1


colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]

endpts = list(np.linspace(0.001, np.max(out), len(colorscale) - 1))
fips = df_sample['FIPS'].tolist()
values = out

fig = ff.create_choropleth(
    fips=fips, values=values, scope=['usa'],
    binning_endpoints=endpts, colorscale=colorscale,
    #show_state_data=False,
    #county_outline={'color': 'rgb(15, 15, 55)', 'width': 0.01},
    state_outline={'color': 'rgb(155, 155, 155)', 'width': 0.2},
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='USA by Total Likelihood of Bank Failure',
    legend_title='Total Failure'
)
fig['layout']['dragmode'] = 'pan'
fig['layout']['margin']['b'] = 5
fig['layout']['width'] = fig['layout']['width'] * 2
fig['layout']['height'] = fig['layout']['height'] * 2
fig['layout']['xaxis']['fixedrange'] = False
fig['layout']['yaxis']['fixedrange'] = False
plotly.offline.plot(fig, filename='choropleth_full_usa.html')

kys = list(fig['layout'].keys())
print(kys)
print(list(fig['layout']['xaxis'].keys()))
print(list(fig['layout']['yaxis'].keys()))