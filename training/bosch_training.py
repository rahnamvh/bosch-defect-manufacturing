from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from sklearn.metrics import matthews_corrcoef
import datetime
import time
from pyspark.ml.linalg import Vectors
from collections import Counter

def read_test_train():
    train_num_in_na=spark.read.csv("file:///home/rahnamvm/dssets/bosch/train_numeric.csv", header='true', inferSchema='true')
    print ("Done Reading Training Ds - Count:{}".format(train_num_in_na.count()))
    test_num_in_na=spark.read.csv("file:///home/rahnamvm/dssets/bosch/test_numeric.csv", header='true', inferSchema='true')
    print ("Done Reading Testing Ds - Count:{}".format(test_num_in_na.count ()))
    train_num_in=train_num_in_na.fillna(0)
    test_num_in=test_num_in_na.fillna(0)
    ignore=['Id', 'Response']
    collist=list(x for x in train_num_in.columns if x not in ignore)
    return train_num_in, test_num_in, collist

def train_test_prep(train_num_in, test_num_in, featimp, collistin):
    if(featimp!='na'):
        collist=featimp
    else:
        collist=collistin
    assembler=VectorAssembler(inputCols=collist, outputCol='Features')
    train_num_inva_temp=assembler.transform(train_num_in)
    test_num_inva_temp=assembler.transform(test_num_in)
    train_num_inva=train_num_inva_temp.select('Id', 'Response', 'Features')
    test_num_inva=test_num_inva_temp.select('Id', 'Features')
    stringIndexer=StringIndexer(inputCol="Response", outputCol="IndexedResponse")
    modelSI=stringIndexer.fit(train_num_inva)
    train_num_final=modelSI.transform(train_num_inva)
    train_num=train_num_final.drop('Response')
    test_num=test_num_inva
    print("Train Ds - Count:{}".format(train_num.count()))
    print("Test Ds - Count:{}".format(test_num.count()))
    return train_num, test_num, 'Id', 'IndexedResponse', 'ResponsePrediction', 'Features'

def gbt_classifier (labelcol, predcol, featcol, MaxIter, MaxDepth, MaxBins, SubsamplingRate, MaxMemoryInMB, CacheNodeIds, MinInstancesPerNode, Seed):
    gbt=GBTClassifier ()
    gbt.setLabelCol(labelcol).setFeaturesCol(featcol).setPredictionCol(predcol).setMaxIter(MaxIter).setMaxDepth(MaxDepth).setMaxBins(MaxBins).setSubsamplingRate(SubsamplingRate).setMaxMemoryInMB(MaxMemoryInMB).setCacheNodeIds(CacheNodeIds).setMinInstancesPerNode(MinInstancesPerNode).setSeed(Seed)
    return gbt

def get_mcc_score(preddslist):
    label = list(x[0] for x in preddslist)
    pred = list(x[1] for x in preddslist)
    mcc_score = matthews_corrcoef(label, pred)
    return mcc_score

def write_prediction(predictionslist):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    f = open("/home/rahnamvm/tmp/" + sub_file, 'w')
    f.write('Id,Response\n')
    for row in predictionslist:
        str1 = str(row[0]) + ',' + str(int(row[1]))
        str1 += '\n'
        f.write(str1)
    f.close()

def split_train(train_num, test_num, gbtclassfr, idcol, labelcol, predcol, nfld):
    seed1 = 1800009193L
    seed2 = 2800009193L
    (trainds, validds) = train_num.randomSplit([0.8, 0.2], seed1)
    trainlbl1 = trainds.filter(trainds.IndexedResponse == 1)
    trainlbl1.cache()
    print("Label 1 count: {}".format(trainlbl1.count()))
    rng = range(1, nfld + 1)
    spltper = list(1/float(nfld) for r in rng)
    trainspltds = trainds.randomSplit(spltper, seed2)
    predictions = []
    predictionsvalid = []
    preddict = []
    predvalidlcdict = []
    predvalidpcdict = []

    splti = 1
    for train in trainspltds:
        print("Training Split No: {}".format(splti))
        print('Starting time: {} '.format(time.asctime(time.localtime(time.time()))))
        trainl1 = train.union(trainlbl1)
        trainl1.cache()
        ##print("Training Count - {}".format(trainl1.count()))
        model = gbtclassfr.fit(trainl1)
        print("Finished Training Split No: {}".format(splti))
        print("Predicting Valid")
        predictionsvalid.append(model.transform(validds).select(idcol, labelcol, predcol))
        print("Predicting Test")
        predictions.append(model.transform(test_num).select(idcol, predcol))
        print('Ending time: {} '.format(time.asctime(time.localtime(time.time()))))
        trainl1.unpersist()
        splti = splti + 1

    trainlbl1.unpersist()

    splti = 1
    for predvalid in predictionsvalid:
        print("Collecting Valid Split No: {}".format(splti))
        predvalidlcdict.append(dict(predvalid.select(idcol, labelcol).rdd.map(lambda r: (r[0], r[1])).collect()))
        if (splti == 1):
            predvalidpcdict.append(dict(predvalid.select(idcol, predcol).rdd.map(lambda r: (r[0], r[1])).collect()))
        splti = splti + 1


    splti = 1
    for pred in predictions:
        print("Collecting Test Split No: {}".format(splti))
        preddict.append(dict(pred.select(idcol, predcol).rdd.map(lambda r: (r[0], r[1])).collect()))
        splti = splti + 1

    print("Combining Valid")
    predictionsvalidlist = list((Counter(list(d[x] for d in predvalidlcdict)).most_common(1)[0][0], predvalidpcdict[0][x])  for x in predvalidlcdict[0].iterkeys())

    print("Combining Test")
    predictionslist = list((x,Counter(list(d[x] for d in preddict)).most_common(1)[0][0])  for x in preddict[0].iterkeys())
    predictionslist.sort()

    return predictionslist , predictionsvalidlist



spark = SparkSession.builder.appName("bosch-training").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

train_num_in, test_num_in, collist = read_test_train()
train_num, test_num, idcol, labelcol, predcol, featcol = train_test_prep(train_num_in, test_num_in, 'na', collist)
MaxIter, MaxDepth, MaxBins, SubsamplingRate, MaxMemoryInMB, CacheNodeIds, MinInstancesPerNode, Seed = 40, 9, 250, 0.7, 1024, True, 1, 42
gbtclassfr = gbt_classifier(labelcol, predcol, featcol, MaxIter, MaxDepth, MaxBins, SubsamplingRate, MaxMemoryInMB, CacheNodeIds, MinInstancesPerNode, Seed)
nfld = 11
predictionslist, predictionsvalidlist = split_train(train_num, test_num, gbtclassfr, idcol, labelcol, predcol, nfld)
print("Positive Count Predicition: {}, MCC Score: {}".format(sum(list(x[1] for x in predictionslist)), get_mcc_score(predictionsvalidlist)))
write_prediction(predictionslist)

spark.stop()
