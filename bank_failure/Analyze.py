import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly as py
import plotly.graph_objs as go
#from pyzipcode import ZipCodeDatabase


# get LAT LONG from zipcodes:
# zipcode = ZCDB[54115]
# zipcode.latitude, zipcode.longitude

#ZCDB = ZipCodeDatabase()

failed = np.matrix([1,2])

with open('../data/banklist.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            new = np.array([int(row[3]), int(row[5][-2:])])
            failed = np.vstack((failed, new))
            line_count += 1
    print(f'Processed {line_count} lines.')

failed = np.delete(failed, (0), axis=0)

num_years = 3


dataset = pd.read_csv("../data/combined_data_v2.csv")
dataset_list = list(dataset.columns)
print(dataset_list)

x = np.array(dataset['asset'])
data = [go.Histogram(x=x)]

#py.offline.plot(data, filename='basic_histogram.html')

temp = dataset.drop(['rssdhcr', 'fed_rssd','name', 'city', 'zip'], axis=1)
temp['bkclass'] = pd.factorize(temp['bkclass'])[0]
temp['stalp'] = pd.factorize(temp['stalp'])[0]

depends = np.transpose(np.array([temp['cert'], temp['repdte'].apply(lambda x: int(x[-2:]))]))
labels = np.zeros(len(dataset['cert']))


for i in range(len(dataset)):
    idx = np.logical_and(np.array(failed[:, 1] > depends[i, 1]) ,  np.array(failed[:, 1] <= (depends[i, 1] + num_years)))
    labels[i] = int(depends[i, 0] in failed[np.where(idx), 0])

temp = temp.drop(['cert', 'docket', 'repdte'], axis = 1)
filter = (depends[:, 1] < 13)

features = np.column_stack([temp, labels])
features = np.nan_to_num(features)
feature_list = list(temp.columns)

train = features[filter, :]
test = features[np.logical_not(filter), :]


#with open('bank_train.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(train)
#with open('bank_test.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(test)
#
#writeFile.close()


print(sum(features[:, -1]) / len(features[:, -1]))
print(train.shape)
print(np.isnan(train).any())
print(test.shape)
print(np.isnan(test).any())

# Instantiate model with 1000 decision trees
print('making model...')
rf = RandomForestRegressor(n_estimators = 50)
# Train the model on training data
print('fitting model...')
rf.fit(np.delete(train, -1, axis=1), train[:, -1])

print('making predictions...')
predictions = rf.predict(np.delete(test, -1, axis=1))

errors = abs(predictions - test[:, -1]) / len(test[:, -1])
print("total error")
print(sum(errors))

#data = [go.Histogram(x=predictions)]

#py.offline.plot(data, filename='basic_histogram.html')

print("Threshold")
K = sum(train[:,-1]) / len(train[:,-1])
print(K)
print("Guessed Yes Correctly")
a = sum(np.logical_and(predictions >= K, test[:, -1] == 1)) / sum(test[:, -1] == 1)
print(a)
print("Guessed No Correctly")
b = sum(np.logical_and(predictions < K, test[:, -1] == 0)) / sum(test[:, -1] == 0)
print(b)
print("F1 score")
print(2*(a*b)/(a+b))

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];





out_filter = np.equal(dataset["repdte"], "12/31/2017")
pred_filter = np.equal(dataset["repdte"][np.logical_not(filter)], "12/31/2017")
testt = np.column_stack([dataset["name"][out_filter], dataset["zip"][out_filter], predictions[pred_filter]])
testt = np.column_stack([testt, testt[:, -1] > K])

print(testt)

np.savetxt("predictions.csv", testt, delimiter=',', header="name,zip,p,prediction", comments="", fmt='%s')