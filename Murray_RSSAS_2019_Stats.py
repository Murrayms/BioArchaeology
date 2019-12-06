#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
Michael Murray 2019 RSSAS
Pig Teeth Stats
6/12/2019
"""
import scipy.stats as scs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# In[12]:


"""there are three meso sites, Can assume they are from the same pop?"""
Sludegaard = [18.0,17.0,16.6,15.7,18.0,17.2,15.5,15.9,16.2,15.6]
Bloksbjerg = [17.5,18.3,16.9,16.8,18.2,14.6,14.5,16.7,17.3]
Nivaa = [17.0,18.6,17.3,16.5]


# In[13]:


boxplotdata = [Sludegaard, Bloksbjerg, Nivaa]
plt.boxplot(boxplotdata,notch = True , labels=['Sludegaard','Bloksbjerg','Nivaa'])
plt.show()


# In[14]:


bins = np.linspace(15, 19, 20)

plt.hist(Sludegaard, bins, alpha=0.5, label='Sludegaard')
plt.hist(Bloksbjerg, bins, alpha=0.5, label='Bloksbjerg')
plt.hist(Nivaa, bins, alpha=0.5, label='Nivaa')

plt.legend(loc='lower right')
plt.xlabel('MTL in mm')
plt.ylabel('Occurrences')

plt.show()


# In[15]:


"""Testing the Mesolithics
first normality 
we're using sapiro since these are smaller samples baring chi^2 and shapiro is unconstrained by sample size"""

print("Sludegaard")
print(scs.shapiro(Sludegaard))
print()
print("Bloksbjerg")
print(scs.shapiro(Bloksbjerg))
print()
print("Nivaa")
print(scs.shapiro(Nivaa))
print()
print("Can not reject null hypothosis of normality for any of these samples")


# In[17]:


"""having checked for abnormality we now can test for equal mean.
since here we believe these samples to have been taken from the same population 
we are using a standard students T test (no concern of different variations)
we'll do a round robin"""

print("Sludegaard & Nivaa")
print(scs.ttest_ind(Sludegaard, Nivaa, equal_var=True))
print()
print("Bloksbjerg & Sludegaard")
print(scs.ttest_ind(Bloksbjerg, Sludegaard, equal_var=True))
print()
print("Nivaa & Bloksbjerg")
print(scs.ttest_ind(Nivaa, Bloksbjerg, equal_var=True))
print()
print("Can not reject null equal mean of normality for any of these samples")


# In[23]:


"""so we can't prove they don't share a mean and we can't proove they arn't all part of the same dist.
do we need to show that we can't proove they arn't part of the same dist? why not"""

kStatA,KpA = scs.ks_2samp(Sludegaard, Nivaa)
print("Sludegaard & Nivaa")
print(kStatA)
print(KpA)
print()

kStatB,KpB = scs.ks_2samp(Bloksbjerg, Sludegaard)
print("Bloksbjerg & Sludegaard")
print(kStatB)
print(KpB)
print()

kStatC,KpC = scs.ks_2samp(Nivaa, Bloksbjerg)
print("Nivaa & Bloksbjerg")
print(kStatC)
print(KpC)

print("In a round robin of KS test we have failed to proove")
print("that these samples are not part of the same distribution")


# In[25]:


mesoLith = Sludegaard+Bloksbjerg+Nivaa
print("We can not say that these samples all represnt the same population of pigs")
print("that being said having failed to prove that they are different we shall not merge them into")
print("a combined mesolithic populaiton to compare with out neolithic population")
boxplotdata = [Sludegaard, Bloksbjerg, Nivaa, mesoLith]
plt.boxplot(boxplotdata,notch = True , labels=['Sludegaard','Bloksbjerg','Nivaa','Combined'])
plt.show()


# In[121]:


"""Combinding the mesolithic site for comparison with the Neolithic"""
neoLith = [13.0,13.2,13.2,13.3,13.5,13.5,13.6,13.6,13.6,13.7,13.8,13.8,13.9,13.9,13.9,13.9,14.0,14.0,14.0,14.0,14.1,14.1,14.2,14.2,14.2,14.2,14.2,14.2,14.2,14.3,14.3,14.4,14.5,14.6,14.6,14.6,14.7,14.9,15.0,15.0,15.1,15.1,15.2,15.3,15.4,15.6,16.1]

neo = pd.DataFrame(neoLith)
meso = pd.DataFrame(mesoLith)


# In[122]:


"""Summary Statistics"""

print("Neolithic Summary Stats:")
print(neo.describe())

print()

print("Mesolithic Summary Stats:")
print(meso.describe())


# In[124]:


"""BoxPlot"""

boxplotdata = [neoLith, mesoLith]
plt.boxplot(boxplotdata,notch = True , labels=['Neolithic Pigs','Mesolithic Pigs'])
plt.show()


# In[125]:


plt.hist(neoLith, 15)
plt.xlabel('Neolithic MTL')
plt.ylabel('Occurrences')
plt.show


# In[126]:


plt.hist(mesoLith, 12)
plt.xlabel('Mesolithic MTL')
plt.ylabel('Occurrences')
plt.show


# In[127]:


"""Test for Normality in our distributions
Returns

    statisticfloat or array

        s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
    pvaluefloat or array

        A 2-sided chi squared probability for the hypothesis test.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
"""

print("Neolithic")
print(scs.normaltest(neoLith))
print("can not reject hypothosis of normality @ 23%")
print()
print("Mesolithic")
print(scs.normaltest(mesoLith))
print("can not reject hypothosis of normality @ 67%")


# In[128]:


bins = np.linspace(13, 19, 18)

plt.hist(neoLith, bins, alpha=0.5, label='Neolithic')
plt.hist(mesoLith, bins, alpha=0.5, label='Mesolithic')
plt.legend(loc='upper right')
plt.xlabel('MTL in mm')
plt.ylabel('Occurrences')

plt.show()


# In[129]:


"""Welch's T test for equal mean becasue equal_var=False

welches because we have significantly different stdevs
 Delacre, M., Lakens, D., & Leys, C. (2017). 
 Why Psychologists Should by Default Use Welch’s t-test Instead of Student’s t-test. 
 International Review of Social Psychology, 30(1), 92–101. DOI: http://doi.org/10.5334/irsp.82 
"""

print(scs.ttest_ind(neoLith, mesoLith, equal_var=False))

print("Extremely unlikely these sample represent populaitons of the same mean")


# In[130]:


"""Two Sample Kolmogorov-Smirnov Test"""

kStat,Kp = scs.ks_2samp(neoLith, mesoLith)

print(kStat)
print(Kp)
print("""Extremely unlikely these samples are taken from the same distribution""")


# In[ ]:




