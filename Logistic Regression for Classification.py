#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn --upgrade --quiet')


# In[2]:


get_ipython().system('pip install opendatasets --upgrade --quiet')


# In[3]:


import opendatasets as od


# In[4]:


od.version()


# In[5]:


dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'


# In[6]:


od.download(dataset_url)


# In[7]:


import os


# In[8]:


data_dir = './weather-dataset-rattle-package'


# In[9]:


os.listdir(data_dir)


# In[10]:


train_csv = data_dir + '/weatherAUS.csv'


# In[11]:


get_ipython().system('pip install pandas --quiet')


# In[12]:


import pandas as pd


# In[13]:


raw_df = pd.read_csv(train_csv)


# In[14]:


raw_df


# In[15]:


raw_df.info()


# In[16]:


raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)


# In[17]:


raw_df.info()


# In[18]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[19]:


raw_df.Location.nunique()


# In[20]:


px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday')


# In[21]:


px.histogram(raw_df, 
             x='Temp3pm', 
             title='Temperature at 3 pm vs. Rain Tomorrow', 
             color='RainTomorrow')


# In[22]:


px.histogram(raw_df, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today')


# In[23]:


px.scatter(raw_df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday')


# In[24]:


px.scatter(raw_df.sample(2000), 
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')


# # (Optional) Working with a Sample
# When working with massive datasets containing millions of rows, it's a good idea to work with a sample initially, to quickly set up your model training notebook. If you'd like to work with a sample, just set the value of use_sample to True.
# 
# use_sample = False
# sample_fraction = 0.1
# if use_sample:
#     raw_df = raw_df.sample(frac=sample_fraction).copy()

# In[25]:


get_ipython().system('pip install scikit-learn --upgrade --quiet')


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[28]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[29]:


plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);


# In[30]:


year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


# In[31]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# # Identifying Input and Target Columns
# 
# Often, not all the columns in a dataset are useful for training a model. In the current dataset, we can ignore the `Date` column, since we only want to weather conditions to make a prediction about whether it will rain the next day.
# 
# Let's create a list of input columns, and also identify the target column.

# In[32]:


raw_df


# In[33]:


input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'


# input_cols

# In[34]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[35]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()


# In[36]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# Let's also identify which of the columns are numerical and which ones are categorical. This will be useful later, as we'll need to convert the categorical data to numbers for training a logistic regression model.

# In[37]:


get_ipython().system('pip install numpy --quiet')


# In[38]:


import numpy as np


# In[39]:


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[40]:


train_inputs[numeric_cols].describe()


# In[41]:


train_inputs[categorical_cols].nunique()


# ## Imputing Missing Numeric Data
# 
# Machine learning models can't work with missing numerical data. The process of filling missing values is called imputation.
# 
# <img src="https://i.imgur.com/W7cfyOp.png" width="480">
# 
# There are several techniques for imputation, but we'll use the most basic one: replacing missing values with the average value in the column using the `SimpleImputer` class from `sklearn.impute`.

# In[42]:


from sklearn.impute import SimpleImputer


# In[43]:


imputer = SimpleImputer(strategy = 'mean')


# In[44]:


train_inputs[numeric_cols].isna().sum()


# In[45]:


imputer.fit(raw_df[numeric_cols])


# In[46]:


list(imputer.statistics_)


# In[47]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[48]:


train_inputs[numeric_cols].isna().sum()


# ## Scaling Numeric Features
# 
# Another good practice is to scale numeric features to a small range of values e.g. $(0,1)$ or $(-1,1)$. Scaling numeric features ensures that no particular feature has a disproportionate impact on the model's loss. Optimization algorithms also work better in practice with smaller numbers.
# 
# The numeric columns in our dataset have varying ranges.

# Let's use `MinMaxScaler` from `sklearn.preprocessing` to scale values to the $(0,1)$ range.

# In[49]:


from sklearn.preprocessing import MinMaxScaler


# In[50]:


scaler = MinMaxScaler()


# In[51]:


scaler.fit(raw_df[numeric_cols])


# In[52]:


print('Minimum:')
list(scaler.data_min_)


# In[53]:


print('Maximum:')
list(scaler.data_max_)


# In[54]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# In[55]:


train_inputs[numeric_cols].describe()


# ## Encoding Categorical Data
# 
# Since machine learning models can only be trained with numeric data, we need to convert categorical data to numbers. A common technique is to use one-hot encoding for categorical columns.
# 
# <img src="https://i.imgur.com/n8GuiOO.png" width="640">
# 
# One hot encoding involves adding a new binary (0/1) column for each unique category of a categorical column. 

# We can perform one hot encoding using the `OneHotEncoder` class from `sklearn.preprocessing`.

# In[56]:


from sklearn.preprocessing import OneHotEncoder


# In[57]:


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# In[58]:


encoder.fit(raw_df[categorical_cols])


# In[59]:


encoder.categories_


# In[60]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


# In[61]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# In[62]:


pd.set_option('display.max_columns', None)


# In[63]:


test_inputs


# ## Saving Processed Data to Disk
# 
# It can be useful to save processed data to disk, especially for really large datasets, to avoid repeating the preprocessing steps every time you start the Jupyter notebook. The parquet format is a fast and efficient format for saving and loading Pandas dataframes.

# In[64]:


print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)


# In[65]:


get_ipython().system('pip install pyarrow --quiet')


# In[66]:


train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')


# In[67]:


pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')


# In[68]:


train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')

train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]


# ## Training a Logistic Regression Model
# 
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# 
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# 
# 
# Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):
# 
# 
# <img src="https://i.imgur.com/YMaMo5D.png" width="480">
# 
# The sigmoid function applied to the linear combination of inputs has the following formula:
# 
# <img src="https://i.imgur.com/sAVwvZP.png" width="400">
# 
# To train a logistic regression model, we can use the `LogisticRegression` class from Scikit-learn.

# In[69]:


from sklearn.linear_model import LogisticRegression


# In[70]:


model = LogisticRegression(solver = 'liblinear')


# In[71]:


get_ipython().run_line_magic('pinfo', 'LogisticRegression')


# In[72]:


model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# `model.fit` uses the following workflow for training the model ([source](https://www.deepnetts.com/blog/from-basic-machine-learning-to-deep-learning-in-5-minutes.html)):
# 
# 1. We initialize a model with random parameters (weights & biases).
# 2. We pass some inputs into the model to obtain predictions.
# 3. We compare the model's predictions with the actual targets using the loss function.  
# 4. We use an optimization technique (like least squares, gradient descent etc.) to reduce the loss by adjusting the weights & biases of the model
# 5. We repeat steps 1 to 4 till the predictions from the model are good enough.
# 
# 
# <img src="https://i.imgur.com/g32CoIy.png" width="480">
# 
# For a mathematical discussion of logistic regression, sigmoid activation and cross entropy, check out [this YouTube playlist](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=1). Logistic regression can also be applied to multi-class classification problems, with a few modifications.

# In[73]:


print(numeric_cols + encoded_cols)


# In[74]:


print(model.coef_)


# In[75]:


weight_df = pd.DataFrame({
    'feature' : (numeric_cols + encoded_cols),
    'weight' : model.coef_.tolist()[0]
})
weight_df


# In[76]:


print(model.intercept_)


# Matplotlib and Seaborn can be used together seamlessly because Seaborn is built on top of Matplotlib. 

# In[77]:


plt.figure(figsize=(10, 50))
sns.barplot(data=weight_df, x='weight', y='feature')


# In[78]:


sns.barplot(data=weight_df.sort_values('weight', ascending = False).head(10), x='weight', y='feature')


# In[79]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[80]:


train_preds = model.predict(X_train)
train_preds


# In[81]:


train_targets


# We can output a probabilistic prediction using predict_proba.
# 
# 

# In[82]:


train_probs = model.predict_proba(X_train)
train_probs


# In[83]:


model.classes_


# We can test the accuracy of the model's predictions by computing the percentage of matching values in `train_preds` and `train_targets`.
# 
# This can be done using the `accuracy_score` function from `sklearn.metrics`.

# In[84]:


from sklearn.metrics import accuracy_score


# In[85]:


accuracy_score(train_preds, train_targets)


# The model achieves an accuracy of 85.1% on the training set. We can visualize the breakdown of correctly and incorrectly classified inputs using a confusion matrix.
# 
# <img src="https://i.imgur.com/UM28BCN.png" width="480">

# In[86]:


from sklearn.metrics import confusion_matrix


# In[87]:


confusion_matrix(train_targets, train_preds, normalize='true')


# Let's define a helper function to generate predictions, compute the accuracy score and plot a confusion matrix for a given st of inputs.

# In[88]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));
    
    return preds


# In[89]:


train_preds = predict_and_plot(X_train, train_targets, 'Training')


# In[90]:


val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')


# In[91]:


test_preds = predict_and_plot(X_test, test_targets, 'Test')


# In[92]:


def random_guess(inputs):
    return np.random.choice(["No","Yes"], len(inputs))


# In[93]:


def all_no(inputs):
    return np.full("No",len(inputs))


# In[94]:


accuracy_score(test_targets, random_guess(X_test))


# ## Making Predictions on a Single Input
# 
# Once the model has been trained to a satisfactory accuracy, it can be used to make predictions on new data. Consider the following dictionary containing data collected from the Katherine weather department today.

# In[95]:


new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[96]:


new_input_df = pd.DataFrame([new_input])


# In[97]:


new_input_df


# In[98]:


new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


# In[99]:


X_new_input = new_input_df[numeric_cols + encoded_cols]


# In[100]:


prediction = model.predict(X_new_input)[0]
prediction


# In[101]:


prob = model.predict_proba(X_new_input)[0]
prob


# # Define a helper function to make predictions for individual inputs.

# In[102]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


# In[103]:


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[104]:


predict_input(new_input)


# ## Saving and Loading Trained Models
# 
# We can save the parameters (weights and biases) of our trained model to disk, so that we needn't retrain the model from scratch each time we wish to use it. Along with the model, it's also important to save imputers, scalers, encoders and even column names. Anything that will be required while generating predictions using the model should be saved.
# 
# We can use the `joblib` module to save and load Python objects on the disk. 

# In[105]:


import joblib


# In[106]:


aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[107]:


joblib.dump(aussie_rain, 'aussie_rain.joblib')


# The object can be loaded back using `joblib.load`

# In[108]:


aussie_rain2 = joblib.load('aussie_rain.joblib')


# In[122]:


aussie_rain2['model'].coef_


# In[123]:


test_preds2 = aussie_rain2['model'].predict(X_test)
accuracy_score(test_targets, test_preds2)


# In[ ]:




