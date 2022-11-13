#!/usr/bin/env python
# coding: utf-8

# ## Restaurant Recommendation Engine

# ##### Importing libraries and loading dataset

# In[11]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


restaurant_data=pd.read_csv(r'C:\Users\vibha\Desktop\Data Science Projects\Restaurant Recommendation Engine\archive\TripAdvisor_RestauarantRecommendation.csv')
print(restaurant_data.head())


# #### Selecting the required columns

# In[10]:


restaurant_data=restaurant_data[['Name', 'Street Address', 'Location', 'Type']]
restaurant_data=restaurant_data.dropna()
print(restaurant_data)


# #### Using the Type column as the feature to recommend similar restaurants to the customer:

# In[43]:


tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(restaurant_data['Type'].astype(str))
similarity = cosine_similarity(tfidf_matrix)
print(similarity)


# #### Setting the name of the restaurant as an index so that we can find similar restaurants by giving the name of the restaurant as an input

# In[48]:


indices = pd.Series(restaurant_data.index, index=restaurant_data['Name'])
print(indices)


# #### Function for to recommend a similar restaurant

# In[50]:


def restaurant_recommendation(name, similarity = similarity):
    index = indices[name]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    restaurantindices = [i[0] for i in similarity_scores]
    return restaurant_data['Name'].iloc[restaurantindices]

print(restaurant_recommendation("Malbec"))

