import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
import itertools
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class CSV_Reader:



  def read(self):
      df = None
      try:
          df_dataset = pd.read_csv('combined_data_v2.csv')
          df_failed_banks = pd.read_csv('banklist.csv')
      except:
          df = None
          print ('Unable to read CSV and error occurred')
      df_dataset.reindex_axis(sorted(df_dataset.columns), axis=1)
      return df_dataset,df_failed_banks
  def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month



class Classifier:

    def __init__(self,dataset,ratio = 0.7):

        np.random.shuffle(dataset)
        self.ratio = ratio
        y = dataset[:,dataset.shape[1]-1]

        x = dataset[:,2:dataset.shape[1]-1]
        certs = dataset[:,0]

        y = np.array(y).astype(np.float)
        trainingSetSize = int(len(x)*self.ratio)
        testSetSize = len(x)-trainingSetSize
        self.trainingSet = x[0:trainingSetSize]
        self.yTrain = y[0:trainingSetSize]
        self.yTest = y[trainingSetSize:len(x)]
        self.testSet = x[trainingSetSize:len(x)]
        self.train_cert = certs[0:trainingSetSize]
        self.test_cert = certs[trainingSetSize:len(x)]





    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    def gridFitAdaBoost(self,trainingSet,testSet,yTrain,yTest):

        from sklearn.ensemble import RandomForestClassifier
        dt = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=5,max_depth=10)#,min_samples_split=10,max_depth=8)

        clf = AdaBoostClassifier(dt,n_estimators=10)

        #clf = RandomForestClassifier(n_estimators=10)

        start_tr_time = time.time()

        dtc, p,scaler = self.stratified_kfold_fitted(trainingSet, yTrain, 10, clf,scaled=False)
        stop_tr_time = time.time() - start_tr_time

        print ('cv f1 ' + str(p.mean()))
        start_pr_time = time.time()

        predicted = dtc.predict(testSet)
        stop_pr_time = start_pr_time-time.time()

        accuracyScore = f1_score(yTest, predicted, average='micro' )

        print ('f1 ' + str(accuracyScore))
        # dt = OneVsRestClassifier(estimator=dt1)
        start_gs_time = time.time()

       # clf, score = kfold_fitted(trainingSet,yTrain,5,dt1)

        class_names = ['not failed','failed']

        cfm = confusion_matrix(yTest, predicted, labels=range(len(class_names)))
        #auc_score = roc_auc_score(yTest, predicted)
        self.plot_confusion_matrix(cfm, classes=class_names, normalize=True,title='Normalized confusion matrix')

        print ('Training time ' + str(stop_tr_time))
        print ('Prediction time ' + str(stop_pr_time))

        return dtc




    def stratified_kfold_fitted(self,x_train, y_train, folds, clf,scaled = False):
        scores = []
        scaler = StandardScaler()
        kfold = StratifiedKFold(y=y_train, n_folds=folds, random_state=1, shuffle=True)
        for k, (train, test) in enumerate(kfold):
            tr = x_train[train]
            te = x_train[test]
            if scaled:
                tr = scaler.fit_transform(tr)
                te = scaler.transform(te)
            clf.fit(tr, y_train[train])
            predicted = clf.predict(te)
            score = f1_score(y_train[test], predicted, average='micro')
            scores.append(score)

        scores = np.array(scores).astype(np.float)
        return clf, scores,scaler

    def kfold_fitted(self,x_train,y_train,folds,clf,scaled = False):
        scores = []
        scaler = StandardScaler()
        #kfold = StratifiedKFold(y=y_train,n_folds=folds,random_state=1,shuffle=True)
        kfold = KFold(len(y_train),folds,random_state=True)
        for train,test in  kfold:
            tr = x_train[train]
            te = x_train[test]
            if scaled :
                tr = scaler.fit_transform(tr)
                te = scaler.transform(te)
            clf.fit(tr,y_train[train])
            pred = clf.predict(te)
            score = accuracy_score(pred,y_train[test])
            scores.append(score)

        scores = np.array(scores).astype(np.float)
        return clf,scores,scaler






if __name__ == '__main__':
    csv = CSV_Reader()
    df_dataset, df_failed = csv.read()
    df_dataset.fillna(value=0,inplace=True)

    asset = df_dataset['asset']
    assavg = np.average(asset)
    cert = df_dataset['cert']
    report_date = df_dataset['repdte']
    report_date =  pd.to_datetime(report_date, infer_datetime_format=True)
    ch_balance = df_dataset['chbal']
    ch_balance_interest = df_dataset['chbali']
    perchengate_change_in_net_loans_and_leases = df_dataset['lnlsnet']
    trade = df_dataset['trade']/assavg
    frepo = df_dataset['frepo']
    bkprem = df_dataset['bkprem']/assavg
    intangible = df_dataset['intan']/assavg
    deposit = df_dataset['dep']/assavg
    deposit_interest = df_dataset['depi']/assavg
    total_domestic_deposit = df_dataset['depdom']/assavg
    Liquidity = (df_dataset['frepp']-frepo)/asset
    Forclosure_ratio = df_dataset['ore']/asset
    IENC = df_dataset['oaienc']/asset#Income earned, not collected on loans
    bank_size = np.log(asset)
    WHOF = (df_dataset['frepp']+df_dataset['idobrmtg'])/asset#Wholesale funding over assets
    OL = (df_dataset['tradel']+df_dataset['idoliab'])/asset#Other liabilities over assets
    eqtot = df_dataset['eqtot']
    nclnls = df_dataset['nclnls']#Percent Change in Noncurrent loans and leases
    rwajt = df_dataset['rwajt']# total risk weighted assets adjusted.
    volatility_liabilities = df_dataset['voliab']
    closed_bank_cert = df_failed['CERT']
    closed_date = df_failed['Closing Date']
    closed_date = pd.to_datetime(closed_date, infer_datetime_format=True)

    dataset = []
    for i in range(len(cert)):
        row = []
        row.append(cert.values[i])
        row.append(report_date.values[i])
        row.append(asset.values[i])
        row.append(ch_balance.values[i])
        row.append(ch_balance_interest.values[i])
        row.append(perchengate_change_in_net_loans_and_leases.values[i])
        row.append(trade.values[i])
        row.append(frepo.values[i])
        row.append(bkprem.values[i])
        row.append(intangible.values[i])
        row.append(deposit.values[i])
        row.append(deposit_interest.values[i])
        row.append(total_domestic_deposit.values[i])
        row.append(Liquidity.values[i])
        row.append(Forclosure_ratio.values[i])
        row.append(IENC.values[i])
        row.append(bank_size.values[i])
        row.append(WHOF.values[i])
        row.append(OL.values[i])
        row.append(eqtot.values[i])
        row.append(nclnls.values[i])
        row.append(rwajt.values[i])
        row.append(volatility_liabilities.values[i])
        if cert.values[i] in closed_bank_cert.values.tolist():

            idx = closed_bank_cert.values.tolist().index(cert.values[i])
            z = pd.DataFrame({'a':[closed_date.values[idx]],'b':[report_date.values[i]]})
            td_series = (z['a'] - z['b'])
            months_diff = td_series.astype('timedelta64[M]').astype(int).values[0]

            #print(months_diff)
            if months_diff <= 12:
                row.append(1.0)
            else:
                row.append(0.0)
        else:
            row.append(0.0)
            #row.append(None)

        dataset.append(row)

    dataset = np.asarray(dataset)
    print('Dataset built')

    clf = Classifier(dataset,ratio=0.7)
    x = clf.trainingSet
    y = clf.yTrain

    ada = clf.gridFitAdaBoost(x,clf.testSet,y,clf.yTest)

    x_full = np.vstack((clf.trainingSet,clf.testSet))

    dbscan = DBSCAN(eps=0.3,min_samples=5,metric='euclidean',leaf_size=30)

    dbscan.fit(x_full)
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = x_full[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = x_full[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()








