# Customer Life Time Value using RFM & Machine Learning

![lifeline.png](5cd2af28-2a81-4dde-8fa9-ae6f8509f416.png)

## Table of Content

## Intro

Customers are the life and soul of every business. Insights on customer behaviours and trends could be a crucial aspect in determining the success of any business. 
Customer Lifetime Value is a metric that describes how important or valuable a customer is to the business. CLV is a hollistic understanding of the customer journey with you, from beginning all the way to the end (Unless you can prevent the end!).
Companies use this as a metric to gauge profitability and understand which customers to focus on. Understanding CLV could help understand the profit that is estimated from the future relationship with a customer.

In this project, I will be looking at transactional data from a company called CDNow, a company which was considered a tech giant in the dot-com bubble, before inevitably closing it's doors in 2013.
This data contains transactional data for a cohort who had their first purchasing order between Jan, 1997 and March, 1997, and contains all subsequent transactions from this cohort up until June 30, 1998.
This data falls under the non-contractual type; which means, customers pay for the product and that's all. There are no subscription fees involved or mandatory recurring payments.

In this project, I intend to add value by shedding light on:
1) How much will each customer spend in the future?
2) What is the probability that a specific customer will make another purchase in the future?

To help solve these business questions, I will answer these 3 questions:
1) Which customers have the highest spend probability in the next 90 days? (Identify our whales)
2) Which customers have recently purchased but are unlikely to buy again? (How do we prevent this customer from 'dying'?)
3) Which customers were predicted to purchase but did not? (Missed opportunity)

High-Level Methodology I will use in this project:
1) Clean and prepare the data
2) Temporal (time-based) splitting of the data to create what is essentially a train-test split
3) Feature Engineering: I will be using the RFM (Recency-Frequency-Monetary) features for this model
4) Run the predictive models
5) Provide insights and analysis

## Part 1 Import Libraries


```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

import plotnine as pn
from plotnine import ggplot
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
```

## Part 2 Data Preparation and Overview


```python
# Import the Data

cdnow_raw_df = pd.read_csv("C:/Users/Kinan Touma/Documents/Portfolio Projects/CLV/cdnow.csv", sep = ',')
```


```python
# Check Columns

cdnow_raw_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>customer_id</th>
      <th>date</th>
      <th>quantity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1997-01-01</td>
      <td>1</td>
      <td>11.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1997-01-12</td>
      <td>1</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>1997-01-12</td>
      <td>5</td>
      <td>77.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>1997-01-02</td>
      <td>2</td>
      <td>20.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>1997-03-30</td>
      <td>2</td>
      <td>20.76</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop first column

cdnow_raw_df = cdnow_raw_df.drop(cdnow_raw_df.columns[0], axis = 1)
```


```python
# Check columns again

cdnow_raw_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>date</th>
      <th>quantity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1997-01-01</td>
      <td>1</td>
      <td>11.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1997-01-12</td>
      <td>1</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1997-01-12</td>
      <td>5</td>
      <td>77.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1997-01-02</td>
      <td>2</td>
      <td>20.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1997-03-30</td>
      <td>2</td>
      <td>20.76</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check info to make sure formatting is good

cdnow_raw_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 69659 entries, 0 to 69658
    Data columns (total 4 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   customer_id  69659 non-null  int64  
     1   date         69659 non-null  object 
     2   quantity     69659 non-null  int64  
     3   price        69659 non-null  float64
    dtypes: float64(1), int64(2), object(1)
    memory usage: 2.1+ MB
    


```python
# Convert date to date time

cdnow_df = cdnow_raw_df.assign(date = lambda x: x['date'].astype(str)).assign(date = lambda x: pd.to_datetime(x['date'])).dropna()
```


```python
# Check info again

cdnow_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 69659 entries, 0 to 69658
    Data columns (total 4 columns):
     #   Column       Non-Null Count  Dtype         
    ---  ------       --------------  -----         
     0   customer_id  69659 non-null  int64         
     1   date         69659 non-null  datetime64[ns]
     2   quantity     69659 non-null  int64         
     3   price        69659 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), int64(2)
    memory usage: 2.1 MB
    


```python
# Understanding the Data:
# Start off by getting a customer's first purchase

cdnow_first_purchase_table = cdnow_df.sort_values(['customer_id', 'date']).groupby('customer_id').first()
cdnow_first_purchase_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>quantity</th>
      <th>price</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1997-01-01</td>
      <td>1</td>
      <td>11.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-01-12</td>
      <td>1</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-01-02</td>
      <td>2</td>
      <td>20.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-01-01</td>
      <td>2</td>
      <td>29.33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1997-01-01</td>
      <td>2</td>
      <td>29.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23566</th>
      <td>1997-03-25</td>
      <td>2</td>
      <td>36.00</td>
    </tr>
    <tr>
      <th>23567</th>
      <td>1997-03-25</td>
      <td>1</td>
      <td>20.97</td>
    </tr>
    <tr>
      <th>23568</th>
      <td>1997-03-25</td>
      <td>1</td>
      <td>22.97</td>
    </tr>
    <tr>
      <th>23569</th>
      <td>1997-03-25</td>
      <td>2</td>
      <td>25.74</td>
    </tr>
    <tr>
      <th>23570</th>
      <td>1997-03-25</td>
      <td>3</td>
      <td>51.12</td>
    </tr>
  </tbody>
</table>
<p>23570 rows × 3 columns</p>
</div>




```python
# Visualize all purchases within cohort

individual_plot = cdnow_df.reset_index().set_index('date')[('price')].resample(rule = 'MS').sum()

individual_plot.plot()

plt.savefig('individual plot.jpg', dpi = 300)

plt.show()
```


    
![png](output_16_0.png)
    



```python
# Visualize Individual Customer Purchases

# Select first 10 IDs
ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]

# Create a new df with these 10 IDs to help visualize
cdnow_customer_id_subset_df = cdnow_df[cdnow_df['customer_id'].isin(ids_selected)].groupby(['customer_id', 'date']).sum().reset_index()

# Visualize

pn.ggplot(
    data = cdnow_customer_id_subset_df,
    mapping = pn.aes(x = 'date', y = 'price', group = 'customer_id')
) +pn.geom_line() +pn.geom_point() +pn.facet_wrap('customer_id') +pn.scale_x_date(date_breaks = '1 year', date_labels = '%Y')
```

    C:\Users\Kinan Touma\anaconda3\Lib\site-packages\plotnine\geoms\geom_path.py:113: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    C:\Users\Kinan Touma\anaconda3\Lib\site-packages\plotnine\geoms\geom_path.py:113: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    C:\Users\Kinan Touma\anaconda3\Lib\site-packages\plotnine\geoms\geom_path.py:113: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    


    
![png](output_17_1.png)
    

- We can see that customer 1, 2, 6, and 10 only purchased once and .
- We can see that customers 3, 7, 8 and 9 are buying constantly.
- We can see that customers 4 and 5 started off hot and then seem to have 'died'.
## Part 3 Machine Learning

### Part 3.1 Time Splitting


```python
# We are interested in looking at next 90 days
n_days = 90

# The latest purchase in the entire data set
max_date = cdnow_df['date'].max()

# Cutoff will be the max date minus 90 days. This will allow us to use the last 90 days as our test split
cutoff = max_date - pd.to_timedelta(n_days, unit = 'd')

# Time-in-data will be our 'Train' and time-out-data will be our 'Test'
time_in_df = cdnow_df[cdnow_df['date'] <= cutoff]
time_out_df = cdnow_df[cdnow_df['date'] > cutoff]
```

### Part 3.2 Feature Engineering


```python
# Make targets from the out-data

targets_df = time_out_df \
    .drop(['quantity', 'date'], axis = 1) \
    .groupby('customer_id') \
    .sum() \
    .rename({'price': 'spend_90_total'}, axis = 1) \
    .assign(spend_90_flag = 1)
```


```python
# Make a Recency Feature from in-data

max_date = time_in_df['date'].max()

recency_df = time_in_df[['customer_id', 'date']] \
    .groupby('customer_id') \
    .apply(lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, 'd')) \
    .to_frame() \
    .set_axis(['recency'], axis  = 1)           
```


```python
# Make Frequency Feature from in-data

frequency_df = time_in_df[['customer_id', 'date']] \
    .groupby('customer_id') \
    .count() \
    .set_axis(['frequency'], axis = 1)
```


```python
# Make Monetary Feature from in-data

monetary_df = time_in_df \
    .groupby('customer_id') \
    .aggregate({'price':['sum', 'mean']}) \
    .set_axis(['price_sum', 'price_mean'], axis = 1)
```


```python
# Combine Features into a dataframe

features_df = pd.concat(
    [recency_df, frequency_df, monetary_df], axis = 1).merge(targets_df, left_index = True, right_index = True, how = 'left').fillna(0)
```


```python
features_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-455.0</td>
      <td>1</td>
      <td>11.77</td>
      <td>11.770000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-444.0</td>
      <td>2</td>
      <td>89.00</td>
      <td>44.500000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-127.0</td>
      <td>5</td>
      <td>139.47</td>
      <td>27.894000</td>
      <td>16.99</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-110.0</td>
      <td>4</td>
      <td>100.50</td>
      <td>25.125000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-88.0</td>
      <td>11</td>
      <td>385.61</td>
      <td>35.055455</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23566</th>
      <td>-372.0</td>
      <td>1</td>
      <td>36.00</td>
      <td>36.000000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23567</th>
      <td>-372.0</td>
      <td>1</td>
      <td>20.97</td>
      <td>20.970000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23568</th>
      <td>-344.0</td>
      <td>3</td>
      <td>121.70</td>
      <td>40.566667</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23569</th>
      <td>-372.0</td>
      <td>1</td>
      <td>25.74</td>
      <td>25.740000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23570</th>
      <td>-371.0</td>
      <td>2</td>
      <td>94.08</td>
      <td>47.040000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>23570 rows × 6 columns</p>
</div>



### Part 3.3 Model Building


```python
# Next 90-day Spend Prediction

# Set features as X - these are the regressors
X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]

# Set your targets 
y_spend = features_df['spend_90_total']

# Create regression spec using squared error
xgb_reg_spec = XGBRegressor(objective = "reg:squarederror", random_state = 123)

# Create model using GridSearchCV with varying learning rates. Calculate MAE for each learning rate. Use 5-fold randomized cross-validation. This will split the data into 80-20 split 5 times per parameter and create 5 models per parameter. 
# Once all models are done, it will find out which parameter did best and it will fit a 6th model (refit = True)
xgb_reg_model = GridSearchCV(
    estimator = xgb_reg_spec,
    param_grid = dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ) ,
    scoring = 'neg_mean_absolute_error',
    refit = True,
    cv = 5
)

# Fit the model
xgb_reg_model.fit(X, y_spend)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=XGBRegressor(base_score=None, booster=None,
                                    callbacks=None, colsample_bylevel=None,
                                    colsample_bynode=None,
                                    colsample_bytree=None, device=None,
                                    early_stopping_rounds=None,
                                    enable_categorical=False, eval_metric=None,
                                    feature_types=None, gamma=None,
                                    grow_policy=None, importance_type=None,
                                    interaction_constraints=None,
                                    learning_rate=None, max_bin=None,
                                    max_cat_threshold=None,
                                    max_cat_to_onehot=None, max_delta_step=None,
                                    max_depth=None, max_leaves=None,
                                    min_child_weight=None, missing=nan,
                                    monotone_constraints=None,
                                    multi_strategy=None, n_estimators=None,
                                    n_jobs=None, num_parallel_tree=None,
                                    random_state=123, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.3, 0.5]},
             scoring=&#x27;neg_mean_absolute_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=XGBRegressor(base_score=None, booster=None,
                                    callbacks=None, colsample_bylevel=None,
                                    colsample_bynode=None,
                                    colsample_bytree=None, device=None,
                                    early_stopping_rounds=None,
                                    enable_categorical=False, eval_metric=None,
                                    feature_types=None, gamma=None,
                                    grow_policy=None, importance_type=None,
                                    interaction_constraints=None,
                                    learning_rate=None, max_bin=None,
                                    max_cat_threshold=None,
                                    max_cat_to_onehot=None, max_delta_step=None,
                                    max_depth=None, max_leaves=None,
                                    min_child_weight=None, missing=nan,
                                    monotone_constraints=None,
                                    multi_strategy=None, n_estimators=None,
                                    n_jobs=None, num_parallel_tree=None,
                                    random_state=123, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.3, 0.5]},
             scoring=&#x27;neg_mean_absolute_error&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=123, ...)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=123, ...)</pre></div></div></div></div></div></div></div></div></div></div>




```python
# Find the best score

xgb_reg_model.best_score_

# on average, our prediction is off by $10.93 per customer
```




    -10.939501193931802




```python
# Find the best parameter to use

xgb_reg_model.best_params_

# best was learning rate of 0.1
```




    {'learning_rate': 0.1}




```python
# Best estimator

xgb_reg_model.best_estimator_
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=123, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=123, ...)</pre></div></div></div></div></div>




```python
# Now we make predictions

prediction_reg = xgb_reg_model.predict(X)
```


```python
prediction_reg
```




    array([ 0.8516166,  2.0874057, 12.327984 , ...,  4.558194 ,  1.140923 ,
            3.5550523], dtype=float32)




```python
# Spend Probability - Classification

# Set features as X - these are the regressors
X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]

# Set your targets 
y_prob = features_df['spend_90_flag']

# Create classification spec using squared error
xgb_clf_spec = XGBClassifier(objective = "binary:logistic", random_state = 123)

# Create model using GridSearchCV with varying learning rates. Calculate MAE for each learning rate. Use 5-fold randomized cross-validation. This will split the data into 80-20 split 5 times per parameter and create 5 models per parameter. 
# Once all models are done, it will find out which parameter did best and it will fit a 6th model (refit = True)
xgb_clf_model = GridSearchCV(
    estimator = xgb_clf_spec,
    param_grid = dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ) ,
    scoring = 'roc_auc', #area under curve
    refit = True,
    cv = 5
)

# Fit the model
xgb_clf_model.fit(X, y_prob)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None, device=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     grow_policy=None, importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=None, max_bin=None,
                                     max_cat_threshold=None,
                                     max_cat_to_onehot=None,
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     multi_strategy=None, n_estimators=None,
                                     n_jobs=None, num_parallel_tree=None,
                                     random_state=123, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.3, 0.5]},
             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None, device=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     grow_policy=None, importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=None, max_bin=None,
                                     max_cat_threshold=None,
                                     max_cat_to_onehot=None,
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     multi_strategy=None, n_estimators=None,
                                     n_jobs=None, num_parallel_tree=None,
                                     random_state=123, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1, 0.3, 0.5]},
             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=123, ...)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=123, ...)</pre></div></div></div></div></div></div></div></div></div></div>




```python
# Find the best score

xgb_clf_model.best_score_

# on average, our prediction is 84% confident
```




    0.8344802222992829




```python
# Find the best parameter to use

xgb_clf_model.best_params_

# best was learning rate of 0.01
```




    {'learning_rate': 0.01}




```python
# Best estimator

xgb_clf_model.best_estimator_
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.01, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=123, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.01, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=123, ...)</pre></div></div></div></div></div>




```python
prediction_clf = xgb_clf_model.predict_proba(X)
```


```python
prediction_clf
```




    array([[0.91622937, 0.08377061],
           [0.9173875 , 0.08261252],
           [0.7599223 , 0.24007767],
           ...,
           [0.86449254, 0.13550743],
           [0.9068537 , 0.09314632],
           [0.88331866, 0.11668134]], dtype=float32)



### Part 3.4 Feature Importance


```python
# Importance of Features for the Regression

imp_spend_amount_dict = xgb_reg_model.best_estimator_.get_booster().get_score(importance_type = 'gain')

imp_spend_amount_df = pd.DataFrame(
    data = {
        'feature': list(imp_spend_amount_dict.keys()),
        'value': list(imp_spend_amount_dict.values())
    })

imp_spend_amount_df = imp_spend_amount_df.sort_values(by='value', ascending=False).reset_index(drop=True)
```


```python
# Visualize

pn.ggplot(
    data = imp_spend_amount_df,
    mapping = pn.aes('feature', 'value')) + pn.geom_col() + pn.coord_flip()
```


    
![png](output_44_0.png)
    



```python
# Importance of Features for the Classification

imp_spend_prob_dict = xgb_clf_model.best_estimator_.get_booster().get_score(importance_type = 'gain')

imp_spend_prob_df = pd.DataFrame(
    data = {
        'feature': list(imp_spend_prob_dict.keys()),
        'value': list(imp_spend_prob_dict.values())
    })

imp_spend_prob_df = imp_spend_prob_df.sort_values(by='value', ascending=False).reset_index(drop=True)
```


```python
pn.ggplot(
    data = imp_spend_prob_df,
    mapping = pn.aes('feature', 'value')) + pn.geom_col() + pn.coord_flip()
```


    
![png](output_46_0.png)
    



```python
# Save Predictions

predictions_df = pd.concat(
    [pd.DataFrame(prediction_reg).set_axis(['pred_spend'], axis = 1), 
     pd.DataFrame(prediction_clf)[[1]].set_axis(['pred_prob'], axis = 1), 
     features_df.reset_index()
    ],
    axis = 1
)
predictions_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.851617</td>
      <td>0.083771</td>
      <td>1</td>
      <td>-455.0</td>
      <td>1</td>
      <td>11.77</td>
      <td>11.770000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.087406</td>
      <td>0.082613</td>
      <td>2</td>
      <td>-444.0</td>
      <td>2</td>
      <td>89.00</td>
      <td>44.500000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.327984</td>
      <td>0.240078</td>
      <td>3</td>
      <td>-127.0</td>
      <td>5</td>
      <td>139.47</td>
      <td>27.894000</td>
      <td>16.99</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.697600</td>
      <td>0.237341</td>
      <td>4</td>
      <td>-110.0</td>
      <td>4</td>
      <td>100.50</td>
      <td>25.125000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29.300179</td>
      <td>0.450179</td>
      <td>5</td>
      <td>-88.0</td>
      <td>11</td>
      <td>385.61</td>
      <td>35.055455</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23565</th>
      <td>1.215649</td>
      <td>0.091442</td>
      <td>23566</td>
      <td>-372.0</td>
      <td>1</td>
      <td>36.00</td>
      <td>36.000000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23566</th>
      <td>1.105468</td>
      <td>0.093146</td>
      <td>23567</td>
      <td>-372.0</td>
      <td>1</td>
      <td>20.97</td>
      <td>20.970000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23567</th>
      <td>4.558194</td>
      <td>0.135507</td>
      <td>23568</td>
      <td>-344.0</td>
      <td>3</td>
      <td>121.70</td>
      <td>40.566667</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23568</th>
      <td>1.140923</td>
      <td>0.093146</td>
      <td>23569</td>
      <td>-372.0</td>
      <td>1</td>
      <td>25.74</td>
      <td>25.740000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23569</th>
      <td>3.555052</td>
      <td>0.116681</td>
      <td>23570</td>
      <td>-371.0</td>
      <td>2</td>
      <td>94.08</td>
      <td>47.040000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>23570 rows × 9 columns</p>
</div>




```python
# Save Models

joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')
```




    ['artifacts/xgb_clf_model.pkl']



## Part 4 Analysis

At the beginning of this project, I promised I would answer three questions.

##### Question 1: Which customers have the highest spend probability in the next 90 days? (Identify our whales)


```python
predictions_df.sort_values('pred_prob', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12859</th>
      <td>431.955383</td>
      <td>0.643970</td>
      <td>12860</td>
      <td>-3.0</td>
      <td>30</td>
      <td>1389.08</td>
      <td>46.302667</td>
      <td>457.26</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5476</th>
      <td>83.370262</td>
      <td>0.643970</td>
      <td>5477</td>
      <td>-2.0</td>
      <td>23</td>
      <td>684.21</td>
      <td>29.748261</td>
      <td>98.45</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8034</th>
      <td>317.018890</td>
      <td>0.643970</td>
      <td>8035</td>
      <td>-3.0</td>
      <td>42</td>
      <td>1332.67</td>
      <td>31.730238</td>
      <td>376.24</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15753</th>
      <td>186.177826</td>
      <td>0.643970</td>
      <td>15754</td>
      <td>-2.0</td>
      <td>19</td>
      <td>724.00</td>
      <td>38.105263</td>
      <td>471.72</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>929.121643</td>
      <td>0.643970</td>
      <td>499</td>
      <td>-3.0</td>
      <td>100</td>
      <td>3427.55</td>
      <td>34.275500</td>
      <td>951.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1.833620</td>
      <td>0.082613</td>
      <td>498</td>
      <td>-431.0</td>
      <td>2</td>
      <td>120.13</td>
      <td>60.065000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6315</th>
      <td>0.464132</td>
      <td>0.082613</td>
      <td>6316</td>
      <td>-427.0</td>
      <td>2</td>
      <td>205.90</td>
      <td>102.950000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>494</th>
      <td>1.894441</td>
      <td>0.082613</td>
      <td>495</td>
      <td>-451.0</td>
      <td>2</td>
      <td>72.07</td>
      <td>36.035000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6365</th>
      <td>2.096353</td>
      <td>0.082613</td>
      <td>6366</td>
      <td>-431.0</td>
      <td>2</td>
      <td>92.69</td>
      <td>46.345000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4617</th>
      <td>1.132867</td>
      <td>0.082613</td>
      <td>4618</td>
      <td>-426.0</td>
      <td>2</td>
      <td>352.50</td>
      <td>176.250000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>23570 rows × 9 columns</p>
</div>




```python
# Lets get the top 10 customers based on the probability that they WILL spend

top_cust = predictions_df.nlargest(10, 'pred_prob')

# Graph their predicted spend vs. actual spend

plt.figure(figsize=(10, 6))
plt.scatter(top_cust['pred_spend'], top_cust['spend_90_total'], color='blue', label='Top 10 Customers')
plt.xlabel('Predicted Spend')
plt.ylabel('Actual Spend')
plt.title('Predicted Spend vs Actual Spend for Top 10 Customers')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_53_0.png)
    


As we can see from the graph above, the data for the top 10 customers appears to be very symmetric and linear. This is further proof that our predictions are fairly accurate.

##### Question 2: Which customers have recently purchased but are unlikely to buy again? (How do we prevent this customer from 'dying'?)


```python
predictions_df[predictions_df['recency'] > -90][predictions_df['pred_prob'] < 0.20].sort_values('pred_prob', ascending = False)
```

    C:\Users\Kinan Touma\AppData\Local\Temp\ipykernel_21764\1436244638.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18032</th>
      <td>8.319474</td>
      <td>0.198952</td>
      <td>18033</td>
      <td>-26.0</td>
      <td>2</td>
      <td>30.48</td>
      <td>15.240</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10887</th>
      <td>8.319474</td>
      <td>0.198952</td>
      <td>10888</td>
      <td>-25.0</td>
      <td>2</td>
      <td>34.95</td>
      <td>17.475</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14808</th>
      <td>8.319474</td>
      <td>0.198952</td>
      <td>14809</td>
      <td>-25.0</td>
      <td>2</td>
      <td>26.84</td>
      <td>13.420</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7743</th>
      <td>8.319474</td>
      <td>0.198952</td>
      <td>7744</td>
      <td>-26.0</td>
      <td>2</td>
      <td>27.69</td>
      <td>13.845</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11821</th>
      <td>8.319474</td>
      <td>0.198952</td>
      <td>11822</td>
      <td>-25.0</td>
      <td>2</td>
      <td>30.79</td>
      <td>15.395</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8077</th>
      <td>14.387550</td>
      <td>0.133946</td>
      <td>8078</td>
      <td>-12.0</td>
      <td>2</td>
      <td>327.93</td>
      <td>163.965</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7365</th>
      <td>5.828395</td>
      <td>0.133896</td>
      <td>7366</td>
      <td>-8.0</td>
      <td>2</td>
      <td>214.01</td>
      <td>107.005</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6971</th>
      <td>14.764078</td>
      <td>0.133896</td>
      <td>6972</td>
      <td>-23.0</td>
      <td>2</td>
      <td>191.94</td>
      <td>95.970</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15504</th>
      <td>14.595891</td>
      <td>0.133896</td>
      <td>15505</td>
      <td>-20.0</td>
      <td>2</td>
      <td>215.33</td>
      <td>107.665</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16100</th>
      <td>14.595891</td>
      <td>0.133896</td>
      <td>16101</td>
      <td>-23.0</td>
      <td>2</td>
      <td>213.62</td>
      <td>106.810</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>411 rows × 9 columns</p>
</div>



These customers are ones that made a purchase in the last 90 days but are unlikely to purchase again. Adding these customers to mailing lists that offer promotions, rewards, and incentives to shop would be ideal as these can serve as tactics to prevent 'death'.

##### Question 3: Which customers were predicted to purchase but did not? (Missed opportunity)


```python
predictions_df[predictions_df['spend_90_total'] == 0].sort_values('pred_spend', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9341</th>
      <td>146.135712</td>
      <td>0.539694</td>
      <td>9342</td>
      <td>-37.0</td>
      <td>19</td>
      <td>906.71</td>
      <td>47.721579</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17476</th>
      <td>141.888000</td>
      <td>0.523910</td>
      <td>17477</td>
      <td>-27.0</td>
      <td>17</td>
      <td>1248.14</td>
      <td>73.420000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>140.815475</td>
      <td>0.574759</td>
      <td>33</td>
      <td>-19.0</td>
      <td>25</td>
      <td>1045.47</td>
      <td>41.818800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17522</th>
      <td>117.985970</td>
      <td>0.612479</td>
      <td>17523</td>
      <td>-5.0</td>
      <td>17</td>
      <td>2280.08</td>
      <td>134.122353</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5054</th>
      <td>113.941353</td>
      <td>0.634723</td>
      <td>5055</td>
      <td>-8.0</td>
      <td>20</td>
      <td>797.11</td>
      <td>39.855500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14395</th>
      <td>-0.656993</td>
      <td>0.137006</td>
      <td>14396</td>
      <td>-3.0</td>
      <td>2</td>
      <td>139.48</td>
      <td>69.740000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12558</th>
      <td>-1.798854</td>
      <td>0.194262</td>
      <td>12559</td>
      <td>-276.0</td>
      <td>17</td>
      <td>477.39</td>
      <td>28.081765</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16451</th>
      <td>-2.658624</td>
      <td>0.261991</td>
      <td>16452</td>
      <td>-140.0</td>
      <td>4</td>
      <td>804.46</td>
      <td>201.115000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3505</th>
      <td>-6.874567</td>
      <td>0.214601</td>
      <td>3506</td>
      <td>-268.0</td>
      <td>24</td>
      <td>1730.57</td>
      <td>72.107083</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5419</th>
      <td>-19.705004</td>
      <td>0.456682</td>
      <td>5420</td>
      <td>-60.0</td>
      <td>24</td>
      <td>1943.58</td>
      <td>80.982500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>20269 rows × 9 columns</p>
</div>



These are the customers ordered by how large of a basket they were predicted to check out, but spent 0. These are some of the most important customers to target with mailing lists focused around retention.

##### The Models

Although on paper our models appear to be predicting properly, let's get some statistical validation to that:


```python
xgb_reg_model.best_score_
```




    -10.939501193931802



This value shows that our regression model is on average, predicting $10.93 off from the actual value. While in the grand scheme this looks alright, we should compare it to the mean spend to see how much of a factor it is.


```python
100 - abs((xgb_reg_model.best_score_) / (np.mean(predictions_df['price_sum'])) * 100)
```




    88.72261401592503



Our regression model is predicting at around 88% of the actual spend, which is not so bad afterall

Let's check the classification model


```python
xgb_clf_model.best_score_
```




    0.8344802222992829



Our classification model is performing at around 83% accuracy. There is room for improvement, but still considered decent

## Next Steps

While creating this CLV model is very valuable and can provide a LOT of insight on it's own; it's strength really shows when it is coupled with a forecasting analysis. A forecast using these models will allows the business to prepare and adapt for scenarios they might not have been expecting.

Understanding the feature importance is also key here. 
In terms of the Regression model, which was used to predict how much a customer would spend in their next 90 days, their total spend and the frequency at which they bought were considered the most important features.
In terms of the Classification model, which was used to predict if the customer would spend at all, their frequency and recencu were considered the most important feature.

Based on the company's needs and what they believe they want to work on, using these feature importances can help them zero in their efforts in the right place.


```python

```
