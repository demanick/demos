#!/usr/bin/env python
# coding: utf-8

# # Python Tips, Tricks, & Best Practices
# This notebook walks you through a series of tips, tricks and best practices to help you write more efficeint, more readable and just all around better Python code. The tasks in this notebook are framed in terms familiar to data scientists but the lessons learned can be applied by anyone writing Python code. Furthermore these are not necessarily the "best" way to tackle these tasks, but they are sufficiently better than typical beginner to novice level code.
# 
# In this notebook you will learn about:
# * List/Dictionary comprehension
# * How to write easy-to-use functions
# * Accessing elements from dictionaries using `.get()`
# * Ternary conditionals
# * Iterating through multiple arrays at once with `zip`
# * `args` and `kwargs`
# * Dynamic object initialization with `setattr`
# 
# And more!

# In[15]:


import random

FEATURES_UNI = [random.randrange(100) for _ in range(10)]
FEATURES_MULTI = [(random.randrange(100), random.randrange(100)) for _ in range(10)]
LABELS = [bool(round(random.random())) for _ in range(10)]


# ## TASK: Feature Engineering
# Create a second feature from the numbers in `FEATURES_UNI` by squaring each. Store these features in a list of tuples, the first object of the tuple should be the original feature and the second is the newly created one.

# In[16]:


# Option 1: for loops
x_transform = []
for x in FEATURES_UNI:
    x_transform.append((x, x ** 2))
print(x_transform)
    
# Option 2: list comprehension
x_transform = [(x, x ** 2) for x in FEATURES_UNI]
print(x_transform)


# Repeat the task but only for even numbers

# In[17]:


# Option 1: for loops
x_transform = []
for x in FEATURES_UNI:
    if x % 2 == 0:
        x_transform.append((x, x ** 2))
print(x_transform)
    
# Option 2: list comprehension
x_transform = [(x, x ** 2) for x in FEATURES_UNI if x % 2 == 0]
print(x_transform)


# Repeat the task again but this time, take the square root of even numbers and the square of all others

# In[29]:


# Option 1: for loops
x_transform = []
for x in FEATURES_UNI:
    if x % 2 == 0:
        x_transform.append((x, x ** 0.5))
    else:
        x_transform.append((x, x ** 2))
print(x_transform)
    
# Option 2: list comprehension
x_transform = [(x, x ** 0.5) if x % 2 == 0 else (x, x ** 2) for x in FEATURES_UNI]
print(x_transform)


# In[30]:


# list comprehension can also be done for dictionaries
x_transform_dict = {x[0]: x[1] for x in x_transform}
print(x_transform_dict)


# ## TASK: Feature Transformation
# Write a function that normalizes a list of data

# In[32]:


def norm(a):
    b = []
    try:
        m = sum(a) / len(a)
        s = (sum([(i - m) ** 2 for i in a]) / len(a)) ** 0.5
        for i in a:
            b.append((i - m) / s)
    except:
        return 0
    return b

norm(FEATURES_UNI)


# This works but has a lot of issues:
# * Difficult to read (from the name of the function itself to the name of the variables, wtf is going on)
# * Try/Except wraps too much logic
# * Function returns multiple data types
# * No documentation
# 
# Let's write a better implementation below:

# In[21]:


def normalize(x):
    """
    Normalizes a list of scalar values using a standard scaler
    
    Args:
      x (list of floats): data to be normalized
      
    Returns a list of floats
    """
    # make sure all values are scalars
    try:
        sum(x)
    except TypeError:
        raise TypeError('All values in list must be either float or int')
    
    # calculate mean and standard deviation
    mean = sum(x) / len(x)
    variance = sum([(val - mean) ** 2 for val in x]) / len(x)
    std = variance ** 0.5
    
    return [(val - mean) / std for val in x]
    
normalize(FEATURES_UNI)


# ## TASK: Weight Initializing
# Given a dictionary of hyperparameters, check for the presence of a key named `weights`. If present, use them as your initial model weights, if not present, use all 0s.

# In[22]:


hparams = {
    'learning_rate': 0.001,
    'epochs': 20,
    'batch_size': 500
}


# In[23]:


# Option 1: if/else statements
if 'weights' in hparams:
    weights = hparams['weights']
else:
    weights = [0. for _ in FEATURES_MULTI[0]]
print(weights)

# Option 2: .get()
weights = hparams.get('weights', [0. for _ in FEATURES_MULTI[0]])
print(weights)

# Option 3: ternary conditionals
weights = hparams.get('weights') or [0. for _ in FEATURES_MULTI[0]]
print(weights)


# ## TASK: Prediction
# Given a list of features and a list of weights, calculate the predicted value by multiplying each feature by its corresponding weight.

# In[24]:


weights = [10 * random.random() for _ in FEATURES_UNI]

# Option 1: for loops and indexing
value = 0
for i in range(len(FEATURES_UNI)):
    value += FEATURES_UNI[i] * weights[i]
print(value)
    
# Option 2: for loops and zip
value = 0
for f, w in zip(FEATURES_UNI, weights):
    value += f * w
print(value)

# Option 3: list comprehension and zip
value = sum([f * w for f, w in zip(FEATURES_UNI, weights)])
print(value)


# ## TASK: Model Building
# You are building a super general machine learning model class for other Data Scientists to use. Because this model is super general it should have only three inputs: `features`, `labels` and `hyperparameters`. How would you implement?

# In[25]:


# Option 1: pass hyperparameters as a dictionary
class BaseModel(object):
    def __init__(self, features, labels, hparams):
        self.features = features
        self.labels = labels
        self.hparams = hparams

# Example
class MyModel(BaseModel):
    pass
my_model = MyModel(FEATURES_UNI, LABELS, {'learning_rate': 0.001, 'epochs': 5})
print(my_model.__dict__)


# In[26]:


# Option 2: pass hyperparameters as dictionary and unpack key-value pairs as attributes
class BaseModel(object):
    def __init__(self, features, labels, hparams):
        self.features = features
        self.labels = labels
        
        for k, v in hparams.items():
            setattr(self, k, v)
        
# Example
class MyModel(BaseModel):
    pass
my_model = MyModel(FEATURES_UNI, LABELS, {'learning_rate': 0.001, 'epochs': 5})
print(my_model.__dict__)


# In[27]:


# Option 3: use kwargs and set as attributes
class BaseModel(object):
    def __init__(self, features, labels, **kwargs):
        self.features = features
        self.labels = labels
        
        for k, v  in kwargs.items():
            setattr(self, k, v)

# Example
class MyModel(BaseModel):
    pass
my_model = MyModel(FEATURES_UNI, LABELS, learning_rate=0.001, epochs=5)
print(my_model.__dict__)


# In[28]:


# Option 4: abandond all sense of structure
class BaseModel(object):
    def __init__(self, *args, **kwargs):
        self.features = args[0]
        self.labels = args[1]
        
        for k, v  in kwargs.items():
            setattr(self, k, v)

# Example
class MyModel(BaseModel):
    pass
my_model = MyModel(FEATURES_UNI, LABELS, ['random', 'shit'], learning_rate=0.001, epochs=5)
print(my_model.__dict__)

