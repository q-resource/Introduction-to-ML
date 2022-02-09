



import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


# In[1]:


# dataset
train_data1 = TabularDataset('train.csv')
train_data=train_data1[:37950]
test_data1 = TabularDataset('train.csv')






test_data_groundtruth=test_data1[37950:]
test_data_groundtruth1=test_data_groundtruth[['Id','Sold Price']]
print(test_data_groundtruth1)#真实数据标签



test_data=test_data_groundtruth.drop(columns=['Sold Price'])


# In[ ]:


id, label = 'Id', 'Sold Price'
large_val_cols = ['Lot', 'Total interior livable area',
                  'Tax assessed value', 'Annual tax amount',
                  'Listed Price', 'Last Sold Price']
for c in large_val_cols + [label]:
    train_data.loc[c] = np.log(train_data[c] + 1)
for c in large_val_cols:
    test_data.loc[c] = np.log(test_data[c] + 1)

# train
predictor = TabularPredictor(label=label,
                             eval_metric='root_mean_squared_error')\
            .fit(train_data.drop(columns=['Zip']),
                 hyperparameters='multimodal', # use multimodal when has GPU
                 num_stack_levels=1,
                num_bag_folds=5)

#predictor=TabularPredictor.load("AutogluonModels/ag-20211107_063744/")
            
# make prediction
predictions = predictor.predict(test_data.drop(columns=['Zip']))

pred = pd.read_csv('sample_submission.csv')
pred[label] = np.exp(predictions) -1
print(pred)
pred.to_csv('autogluonpred.csv', index=False)

