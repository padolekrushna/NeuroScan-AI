#!/usr/bin/env python
# coding: utf-8

# In[3]:


import kagglehub
import shutil
import os


# In[4]:


src_path = kagglehub.dataset_download(
    "masoudnickparvar/brain-tumor-mri-dataset"
)

print("Downloaded at:", src_path)

dst_path = os.path.join(os.getcwd(), "brain-tumor-mri-dataset")

if not os.path.exists(dst_path):
    shutil.move(src_path, dst_path)

print("Dataset moved to:", dst_path)


# In[ ]:




