#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install pandas-profiling --quiet')


# In[8]:


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'


# In[9]:


from urllib.request import urlretrieve


# In[10]:


urlretrieve(medical_charges_url, 'medical.csv')


# In[11]:


import pandas as pd


# In[12]:


medical_df = pd.read_csv('medical.csv')


# In[13]:


medical_df


# In[14]:


medical_df.info()


# In[15]:


medical_df.describe()


# In[16]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[17]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[19]:


fig = px.histogram(medical_df, 
                   x='age', 
                   marginal='box', 
                 
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()


# In[20]:


fig = px.histogram(medical_df, 
                   x='bmi', 
                   marginal='box', 
                   color_discrete_sequence=['red'], 
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()


# In[21]:


fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[22]:


fig = px.histogram(medical_df,
                  x = 'charges',
                  color = 'sex',
                  marginal = 'box',
                  color_discrete_sequence = ['black','green'],
                  title = 'distribution of medical charges in connection with sex')
fig.update_layout(bargap = 0.1)
fig.show()


# In[23]:


fig = px.histogram(medical_df,
                  x = 'charges',
                  color = 'region',
                  marginal = 'box',
                  color_discrete_sequence = ['blue','black'],
                  title = 'distribution of medical charges in connection with region')
fig.update_layout(bargap = 0.1)
fig.show()


# In[24]:


medical_df.smoker.value_counts()


# In[25]:


px.histogram(medical_df,
            x = 'smoker',
            color = 'sex',
            title = 'Smoker')


# In[26]:


px.histogram(medical_df,
            x = 'region',
            color = 'smoker',
            title = 'region and smoker')


# In[27]:


fig = px.scatter(medical_df,
                x ='age',
                y = 'charges',
                color = 'smoker',
                hover_data = ['sex'],
                opacity = 0.8,
                title = 'Age vs Charges')
fig.update_traces(marker_size = 5)
fig.show()


# In[28]:


fig = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[29]:


px.violin(medical_df,
          x = 'children',
          y = 'charges',)


# In[30]:


medical_df.charges.corr(medical_df.age)


# In[31]:


medical_df.charges.corr(medical_df.bmi)


# In[32]:


medical_df.charges.corr(medical_df.children)

# To compute the correlation for categorical columns, they must first be converted into numeric columns.

# In[33]:


smoker_values = {'no' : 0 , 'yes' : 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
smoker_numeric


# In[34]:


medical_df.charges.corr(smoker_numeric)


# In[35]:


medical_df.corr(numeric_only=True)


# In[36]:


px.scatter(medical_df,
          x='age',
          y='age')


# In[37]:


sns.heatmap(medical_df.corr(numeric_only = True), cmap='Reds',annot=True)
plt.title('Correlation Matrix');


# # Linear Regression

# In[38]:


non_smoker_df = medical_df[medical_df.smoker == 'no']


# In[39]:


plt.title('Age vs Charges')
sns.scatterplot(data = non_smoker_df, x = 'age', y = 'charges' , alpha=0.8 , s=15)


# In[40]:


def estimate_charges(age,w,b):
    return w * age + b


# In[41]:


w = 50
b = 100


# In[42]:


estimate_charges(40,w,b)


# In[43]:


ages = non_smoker_df.age
ages


# In[44]:


estimated_charges = estimate_charges(ages,w,b)
estimated_charges


# In[45]:


plt.plot(ages,estimated_charges,'r');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');


# In[46]:


target = non_smoker_df.charges

plt.plot(ages,estimated_charges,'r',alpha=0.9);
plt.scatter(ages,target,s=8,alpha=0.8);
plt.xlabel('Ages');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);


# In[47]:


def try_parameters(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages,w,b)

    plt.plot(ages,estimated_charges, 'r', alpha=0.9);
    plt.scatter(ages,target, s=8 ,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Estimate','Actual']);



# In[48]:


try_parameters(400,5000);


# In[49]:


try_parameters(60, 200)


# # Loss/Cost Function

# In[50]:


predictions = estimated_charges
predictions


# In[51]:


targets = non_smoker_df.charges
targets


# In[52]:


pip install numpy --quiet


# In[53]:


import numpy as np


# In[54]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# In[55]:


w = 50
b = 100


# In[56]:


try_parameters(w, b)


# In[57]:


targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w,b)


# In[58]:


rmse(targets, predicted)


# In[59]:


def try_parameters(w, b):
    age = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b )
    
    plt.plot(ages,predictions, 'r', alpha = 0.9);
    plt.scatter(ages, target, s=8, alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges');
    plt.legend(['Prediction', 'Actual']);
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ",loss)


# In[60]:


try_parameters(50, 100)


# In[61]:


try_parameters(267.24891283,-2091.4205565650827)


# # Linear Regression using Scikit-learn

# In[62]:


get_ipython().system('pip install scikit-learn --quiet')


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


model = LinearRegression()


# In[65]:


help(model.fit)


# In[66]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape : ',inputs.shape)
print('targets shape : ', targets.shape)


# In[67]:


model.fit(inputs,targets)


# In[68]:


model.predict(np.array([[23],
                       [37],
                       [61]]))


# In[69]:


predictions = model.predict(inputs)
predictions


# In[70]:


targets


# In[71]:


rmse(targets,predictions)


# In[72]:


# w
model.coef_


# In[73]:


# b
model.intercept_


# #Use the SGDRegressor class from scikit-learn to train a model using the stochastic gradient descent technique. Make predictions and compute the loss.

# In[74]:


try_parameters(model.coef_,model.intercept_)

from sklearn.linear_model import SGDRegressor


# In[75]:


model1 = SGDRegressor()


# In[76]:


help(model1.fit)


# In[77]:


inputs1 = non_smoker_df[['age']]
targets1 = non_smoker_df.charges


# In[78]:


model1.fit(inputs1,targets1)


# In[79]:


predictions1 = model1.predict(inputs)
predictions1


# In[80]:


rmse(targets,predictions1)


# In[81]:


model1.coef_


# In[82]:


model1.intercept_


# In[83]:


try_parameters(model1.coef_,model1.intercept_)


# #Repeat the steps is this section to train a linear regression model to estimate medical charges for smokers. Visualize the targets and predictions, and compute the loss.

# In[84]:


smoker_df = medical_df[medical_df.smoker == 'yes']


# In[85]:


model2 = LinearRegression()


# In[86]:


inputs2 = smoker_df[['age']]
targets2 = smoker_df.charges


# In[87]:


model2.fit(inputs2,targets2)


# In[88]:


try_parameters(model2.coef_,model2.intercept_)


# In[89]:


inputs2


# In[90]:


targets2


# In[91]:


model.predict(inputs2)


# In[92]:


rmse(model2.coef_,model2.intercept_)


# # As we've seen above, it takes just a few lines of code to train a machine learning model using `scikit-learn`.

# In[93]:


inputs, targets = non_smoker_df[['age']], non_smoker_df['charges']


# In[94]:


model = LinearRegression().fit(inputs,targets)


# In[95]:


predicitons = model.predict(inputs)


# In[96]:


loss = rmse(targets, predictions)


# In[97]:


print('Loss: ',loss)


# # Linear Regression using Multiple Features
# 

# In[98]:


inputs , target = non_smoker_df[['age', 'bmi']] , non_smoker_df['charges']


# In[99]:


model = LinearRegression().fit(inputs,targets)


# In[100]:


predictions = model.predict(inputs)


# In[101]:


loss = rmse(targets,predictions)


# In[102]:


loss

As you can see, adding the BMI doesn't seem to reduce the loss by much, as the BMI has a very weak correlation with charges, especially for non smokers.
# In[103]:


non_smoker_df.charges.corr(non_smoker_df.bmi)


# In[104]:


fig = px.scatter(non_smoker_df, x='bmi', y='charges', title='BMI vs Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[105]:


fig = px.scatter_3d(non_smoker_df, x='age', y='bmi', z='charges')
fig.update_traces(marker_size=3, marker_opacity=0.8)
fig.show()


# # Train a linear regression model to estimate charges using BMI alone. Do you expect it to be better or worse than the previously trained models?

# In[106]:


inputs1,targets1 = non_smoker_df[['bmi']],non_smoker_df['charges']
model = LinearRegression().fit(inputs1,targets1)
predictions = model.predict(inputs1)
loss = rmse(targets1,predictions)
print('rmse: ', loss)
fig = px.scatter(non_smoker_df, x='bmi', y='charges', title='BMI vs Charges')
fig.show()


# In[107]:


non_smoker_df.charges.corr(non_smoker_df.children)


# In[108]:


fig = px.strip(non_smoker_df, x='children', y='charges', title='Children vs Charges')
fig.update_traces(marker_size=4, marker_opacity=0.7)
fig.show()


# In[109]:


inputs , targets = non_smoker_df[['age','bmi','children']], non_smoker_df.charges


# In[110]:


model = LinearRegression().fit(inputs,targets)


# In[111]:


predictions = model.predict(inputs)


# In[112]:


loss = rmse(targets, predictions)
print('Loss: ',loss)


# # Repeat the steps is this section to train a linear regression model to estimate medical charges for smokers. Visualize the targets and predictions, and compute the loss.

# In[113]:


smoker_df = medical_df[medical_df.smoker == "yes"]


# In[114]:


inputs, targets = smoker_df[['age']],smoker_df.charges


# In[115]:


model = LinearRegression().fit(inputs,targets)


# In[116]:


predictions = model.predict(inputs)


# In[117]:


loss = rmse(targets,predictions)
print('Loss: ',loss)


# In[118]:


fig = px.scatter(smoker_df, x='age', y = 'charges')
fig.show()


# In[119]:


inputs, targets = medical_df[['age', 'bmi', 'children']], medical_df.charges


# In[120]:


model = LinearRegression().fit(inputs,targets)


# In[121]:


predictions = model.predict(inputs)


# In[122]:


loss = rmse(targets,predictions)


# In[123]:


print('Loss: ', loss)


# In[124]:


px.scatter(medical_df, x='age', y='charges', color='smoker')


# # Using Categorical Features for Machine Learning

# In[125]:


sns.barplot(data=medical_df, x='smoker', y='charges');


# In[126]:


smoker_codes = {'no':0, 'yes':1}


# In[127]:


medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)


# In[128]:


medical_df.charges.corr(medical_df.smoker_code)


# #new application

# In[129]:


inputs,targets = medical_df[['age', 'bmi', 'children', 'smoker_code']], medical_df.charges


# In[130]:


model = LinearRegression().fit(inputs,targets)


# In[131]:


predictions = model.predict(inputs)


# In[132]:


loss = rmse(targets,predictions)
print('Loss: ',loss)


# # Let's try adding the "sex" column as well.
# 
# 

# In[133]:


sns.barplot(data=medical_df,x='sex', y='charges');


# In[134]:


sex_codes = {'female':0, 'male':1}


# In[135]:


medical_df['sex_code'] = medical_df.sex.map(sex_codes)


# In[136]:


medical_df.charges.corr(medical_df.sex_code)


# In[137]:


inputs,targets = medical_df[['age','bmi','children','smoker_code', 'sex_code']],medical_df.charges
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)
loss = rmse(targets,predictions)
print('Loss= ',loss)


# # One-hot Encoding

# In[138]:


sns.barplot(data=medical_df,x='region',y='charges');


# In[139]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_


# In[140]:


one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot


# In[141]:


medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
medical_df


# In[142]:


# Create inputs and targets
input_cols = ['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
inputs, targets = medical_df[input_cols], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[ ]:





# In[143]:


model.predict([[28,30,2,1,0,0,1,0,0]])


# In[144]:


model.coef_


# In[145]:


weights_df = pd.DataFrame({
    'feature': np.append(input_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df


# # STANDARDIZATION

# In[146]:


from sklearn.preprocessing import StandardScaler


# In[147]:


numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])


# In[148]:


scaler.mean_


# In[149]:


scaler.var_


# In[150]:


medical_df[numeric_cols]


# In[151]:


scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs


# In[155]:


cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values


# In[158]:


inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)

