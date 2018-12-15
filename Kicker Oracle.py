
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import os, json
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz 
import graphviz

from sklearn.metrics import confusion_matrix


# In[2]:


#func that reads the ''category'' column (from JSON format)
def CustomParser1(df):
    j1 = json.loads(df)
    return j1


# This part loads as many files as instructed

# In[3]:


def load_files (n):
    df=pd.DataFrame()
    read_df=pd.DataFrame()
    a=["%03d" % x for x in range(n)]
    for filenum in a:
        filename='Data/Kickstarter'+filenum+'.csv'
        read_df=pd.read_csv(filename,converters={'category':CustomParser1},header=0)
        df=pd.concat([df,read_df],ignore_index=True)
        
    df.info()
    return df
Filesnum=input('How many file should I load? 1..54  ')
df=load_files (int(Filesnum))
df.to_csv('data/jointfile.csv')


# In[4]:


# df=pd.read_csv('Data/Kickstarter.csv',converters={'category':CustomParser1},header=0)
#make differnt columns out of ''category'' format
df[sorted(df['category'][0].keys(),reverse=False)] = df['category'].apply(pd.Series) 
df1=df[['category','color','parent_id','urls','id','name','position']]
df['category.parent_id']=df1['position']
df['category.id']=df1['color']
df['category.position']=df1['parent_id']
df['category.name']=df1['id']
df['category.slug']=df1['name']
#split ''slug'' and leaves just the main category name
df['category.slug']=df['category.slug'].apply(lambda x: x.split('/'))
df['category.slug']=df['category.slug'].apply(lambda x: x.pop(0))
df[['category','category.parent_id','category.id','category.name','category.position','category.slug']][:1]
df.info()


# In[5]:


# func that reads the ''creator'' column (from JSON format).
#some of the cells cause problems 
    #for exemple- the cell JSON format include double apostrophes in nicknames like "Elad "Superman" Toister" confused it.

def CustomParser2(df2):
    try:
        j2 = json.loads(df2)
        return j2
    except: #the func pass all the errored rows and return 0 to the "creator" columnn. 
        return 0
    pass
            
df2=pd.read_csv('data/jointfile.csv',converters={'creator':CustomParser2},header=0)
#count and collect all the droped rows- so we can know the "cost" of te dropping (and maybe i will succed to solve it in the future)
droped=df2.loc[df2['creator']==0,['creator']]
df2=df2.loc[df2['creator']!=0]
drop_list=list(droped.index)
#df['creator'].iloc[drop_list]=df['creator'].iloc[drop_list].apply(lambda x: x.replace(' ',',')) is a start of a solution
df=df.drop(index=drop_list)
print('droped rows:',len(drop_list))
print (len(df))
print (len(df2))


# In[6]:


# 2 func that make diffent columns out of "creator" column (the auto func i used before don't works here. i did it manually)
df2['creator_name']=df2['creator'].apply(lambda x: x['name'])
df2['creator_id']=df2['creator'].apply(lambda x: x['id'])
#"inject" it back to the original df
df['creator_name']=df2['creator_name']
df['creator_id']=df2['creator_id']
df.info()


# In[7]:


#cleaning func
def clean(df):
    data = df.copy()
    #this is important beacuse this is the part we decide  which columns entered the data set.
    #the main structure is like in the exemple but i manipulate and add some additional columns i think we need to include(*marked) . 
    selected_cols = ['creator_name', #*
                     'creator_id', #*
                     'backers_count',
                     'blurb',
                     'is_starred', #*
                     'category.id', #*
                     'category.name',
                     'category.parent_id',
                     'category.slug',
                     'country',
                     'created_at',
                     'currency',  
                     'deadline',
                     'goal',
                     'launched_at',
                     'staff_pick',
                     'state',
                     'usd_pledged',
                     'usd_type']
    data = data[selected_cols]
    data['is_starred']=data['is_starred'].replace({1: True , None: False})
    data = data.dropna()
    successful = data['state'] == "successful"
    failed = data['state'] == "failed"
    cancelled = data['state'] == "cancelled"
    suspended = data['state'] == "suspended"
    data = data.loc[failed | successful | cancelled | suspended]
    num_cols = ['usd_pledged',
                'deadline',
                'created_at',
                'launched_at']
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data['created_at'] = pd.to_datetime(data['created_at'],unit='s')
    data['launched_at'] = pd.to_datetime(data['launched_at'],unit='s')
    data['deadline'] = pd.to_datetime(data['deadline'],unit='s')
    return data

data = clean(df)


# In[8]:


def engineer_features(data):
    #make state 1 or 0
    data['state'].replace('suspended','failed',inplace=True)   
    data['state_num'] = data['state'].apply(lambda x: 1 if x=='successful' else 0)
    #time to reletive time
    data['launched_at_hr'] = data['launched_at'].apply(lambda x: x.hour) + 1
    data['launched_at_day_in_week'] = data['launched_at'].apply(lambda x: x.dayofweek + 1)
    data['launched_at_day_in_month'] = data['launched_at'].apply(lambda x: x.day ) #Elad's comment
    data['launched_at_mo'] = data['launched_at'].apply(lambda x: x.month)
    data['launched_at_yr'] = data['launched_at'].apply(lambda x: x.year)
    data['deadline_hr'] = data['deadline'].apply(lambda x: x.hour) + 1
    data['deadline_day_in_week'] = data['deadline'].apply(lambda x: x.dayofweek + 1) 
    data['deadline_day_in_month'] = data['deadline'].apply(lambda x: x.day ) #Elad's comment
    data['deadline_mo'] = data['deadline'].apply(lambda x: x.month)
    data['deadline_yr'] = data['deadline'].apply(lambda x: x.year)
    data['created_at_hr'] = data['created_at'].apply(lambda x: x.hour) + 1
    data['created_at_day_in_week'] = data['created_at'].apply(lambda x: x.dayofweek + 1) 
    data['created_at_day_in_month'] = data['created_at'].apply(lambda x: x.day )  #Elad's comment
    data['created_at_mo'] = data['created_at'].apply(lambda x: x.month)
    data['created_at_yr'] = data['created_at'].apply(lambda x: x.year)
    data['count'] = 1
    data['success'] = (data['state'] == 'successful')
    data['launched-created'] = (data.launched_at - data.created_at).dt.components.days
    data['deadline-launched'] = (data.deadline - data.launched_at).dt.components.days
    data=data.drop(['launched_at','created_at','deadline'],axis=1) #drop original time col
    data['pledge_perc']=data['usd_pledged']/data['goal']*100
   
    return data
data = engineer_features(data)
data.info()
data.to_csv('Data/data.csv')


# In[9]:


data.to_csv('data/jointfile.csv')


# In[10]:


data=pd.read_csv('data/jointfile.csv')


# In[11]:


corr_df=data.corr()
cor_st=corr_df.loc['state_num']
print(cor_st.nsmallest())
print(cor_st.nlargest())


# # Cut edge

# In[12]:


numrical=[
 'goal',
 'usd_pledged',
 'launched-created',
 'deadline-launched',
 'pledge_perc']


# In[13]:


stats.normaltest(data['goal'])


# In[14]:


data['goal'].describe()
sns.distplot(data["goal"], fit=norm)


# In[15]:


#בהמשך נצטרך לפצל את המידע
goal_highst = data[data["goal"] > 4500] 
goal_lowsat = data[data["goal"] <= 4500] 
log_goal = np.log(data["goal"]) #This data change proves best to fit goal into normal dist.
sqrt_goal = np.sqrt(data["goal"]) 
#sns.distplot(df_highsat[["goal"]], fit=norm)
#sns.distplot(df_lowsat[["goal"]], fit=norm)
sns.distplot(log_goal,bins=None, fit=norm,axlabel='Log_Goal')

#sns.distplot(sqrt_goal, fit=norm)


# In[16]:


#Following investigation update the dataframe for log goal
data['goal']=np.log(data['goal'])


# In[17]:


sns.distplot(data["launched-created"], fit=norm)
data["launched-created"].describe()


# In[18]:


#לחלק שונה בין דגימה של תיאטרון/קולנוע ולא
lu_highst = data[data["launched-created"] >40] 
lu_lowest = data[data["launched-created"] <= 40] 
log_lu = np.log(data["launched-created"])
sqrt_lu = np.sqrt(data["launched-created"])
sns.distplot(lu_highst[["launched-created"]], fit=norm, bins=50)
sns.distplot(lu_lowest[["launched-created"]], fit=norm, bins=50)
#sns.distplot(log_lu, fit=norm)
#sns.distplot(sqrt_lu, fit=norm)


# In[19]:


sns.distplot(data["deadline-launched"], fit=norm)
data["deadline-launched"].describe()


# In[20]:


de_highst = data[data["deadline-launched"] >50] 
de_lowest = data[data["deadline-launched"] <= 50]
#צריך בהמשך לעשות LOG על הגבוה והנמוך בנפרד
log_de = np.log(data["deadline-launched"])
sqrt_de = np.sqrt(data["deadline-launched"])
#sns.distplot(de_highst[["deadline-launched"]], fit=norm)
#sns.distplot(de_lowest[["deadline-launched"]], fit=norm)
#sns.distplot(log_de, fit=norm)
sns.distplot(sqrt_de, fit=norm) 


# In[21]:


#בנתיים נעשה רק SQRT (צריך לבדוק יעילות)
data["deadline-launched"]=np.log(data["deadline-launched"])


# In[22]:


data.to_csv('data/jointfile.csv')  #Save 


# ## Data visualization

# In[23]:


from seaborn import set
plt.rcParams['figure.figsize']=(20,20)
set(font_scale=2)
b=sns.countplot(x='category.slug', hue='success',data=data)
b.set_xlabel("Categories",fontsize=18)
b.set_ylabel("Count",fontsize=18)
b.tick_params(labelsize=14)


# In[24]:


PP=np.clip(data['pledge_perc'], 0, 300)
fig=PP.hist(bins = 200, figsize = (20,15),color='gold')
fig.set_xlabel("% of Capital raised from initial goal",fontsize=10,color='b')
fig.set_ylabel("Count",fontsize=18,color='b')


# In[25]:


corr_matrix = data.corr()
corr_matrix["success"].sort_values()


# In[26]:


sns.heatmap(corr_matrix[(corr_matrix<1) & ((corr_matrix >= 0.2) | (corr_matrix <= -0.2)) ] )


# In[27]:


plt.scatter(x='category.slug', y='usd_pledged', data=data, alpha=0.5, color='r')
plt.ylabel("USD Pledged",fontsize=18)


# In[28]:


sns.factorplot(x='category.slug', y='goal', hue='state_num', kind='bar', data=data, size=15)
locs, labels = plt.xticks();
plt.setp(labels, rotation=90);
plt.title('Range of goal ($) among successful and failed projects')
plt.gca().set_yscale("log", nonposy='clip');
#מראה את הצלחת הפרויקטים לפי סכום היעד לגיוס- רואים שבגדול פרויקטים שנכשלו ביקשו יותר מדי. מצד שני רואים (טבלה הבאה) שפרויקטים שהצליחו לרב יגייסו 150%
#


# In[29]:


df_suc=data.loc[data['success']==True]
ax=sns.factorplot(x='category.slug', y='pledge_perc', kind='bar', data=data, size=15)
locs, labels = plt.xticks();
plt.setp(labels, rotation=90);
plt.title('Pledge % (log) Among Successful Projects')
plt.gca().set_yscale("log", nonposy='clip');
#מה אחוז הגיוס מתוך פריקטים שהצליחו. רואים שניתן לגייס ''פחות'' אבל בפועל יצא יותר


# In[30]:


print('Average pledge % per category')
df_suc.groupby('category.slug', as_index=False, sort=False)['pledge_perc'].mean()


# In[31]:


sns.factorplot(x='category.slug', y='launched-created', hue='state_num', kind='bar', data=data, size=15)
locs, labels = plt.xticks();
plt.setp(labels, rotation=90);
plt.title('Time difference between lunched and created [days]')
plt.gca().set_yscale("linear", nonposy='clip');
#כמעט תמיד עדיף להשיק אחרי שהקמפייין ''מתבשל'' קצת. חריגים הם הקולנוע והתיאטרון


# In[32]:


sns.factorplot(x='category.slug', y='deadline-launched', hue='state_num', kind='bar', data=data, size=7)
locs, labels = plt.xticks();
plt.setp(labels, rotation=90);
plt.title('Deadline-launched [Days]')
plt.gca().set_yscale("linear", nonposy='clip');
#נתון כנראה לא רלוונטי


# #  Data set creation

# In[33]:


data=pd.read_csv('data/jointfile.csv',header=0)


# In[34]:


Dtree_Params=['goal','deadline-launched','launched-created','staff_pick','category.id']
train_df, test_df= train_test_split (data, test_size = 0.2, random_state=6)
Y_train = train_df["state_num"]
X_train = train_df[Dtree_Params]
Y_test=test_df["state_num"]
X_test=test_df[Dtree_Params]


# ## Small Tree ML

# ##  Large Tree ML

# In[35]:



mod1_columns=[ 'is_starred', 'category.name', 'category.slug',
       'country', 'currency', 'goal', 'staff_pick', 'launched_at_hr',
       'launched_at_day_in_week', 'launched_at_day_in_month', 'launched_at_mo',
       'launched_at_yr', 'deadline_hr', 'deadline_day_in_week',
       'deadline_day_in_month', 'deadline_mo', 'deadline_yr', 'created_at_hr',
       'created_at_day_in_week', 'created_at_day_in_month', 'created_at_mo',
       'created_at_yr', 'launched-created',
       'deadline-launched','state_num']

dummies=['category.name', 'category.slug','country', 'currency']

mod1=data[mod1_columns]
mod1 = pd.get_dummies(mod1, columns=dummies) #mod1 will be the first df for the pridiction with spcific columns (maybe afterwords we will wont to add columns)
list(mod1.columns)


# In[36]:


print(len(mod1))
corr_df=mod1.corr()
#corr_df[corr_df['state_num']==1]
#corr_df[(corr_df<1) & ((corr_df >= 0.3) | (corr_df <= -0.3)) ]


# In[37]:


cor_st=corr_df.loc['state_num']
print(cor_st.nsmallest())
print(cor_st.nlargest())


# In[38]:


mod1_params= mod1.columns.drop('state_num')
mod1_params


# In[39]:


Dtree_Params=mod1_params
train_df, test_df= train_test_split (mod1, test_size = 0.2, random_state=6)
Y_train = train_df["state_num"]
X_train = train_df[Dtree_Params]
Y_test=test_df["state_num"]
X_test=test_df[Dtree_Params]
tree_clf = DecisionTreeClassifier (max_depth = 30)
tree_clf.fit (X_train, Y_train)


# In[40]:


# Run the test set through the decision tree
Y2_test_Tree_predict = tree_clf.predict (X_test)  

#Test set conf matrix
conf_matrix_Testdata = confusion_matrix(Y_test, Y2_test_Tree_predict) 
conf_matrix_Testdata


# In[41]:


#Train set conf. matrix
conf_matrix_Traindata = confusion_matrix(Y_train, tree_clf.predict (X_train)) 
conf_matrix_Traindata


# In[42]:


from sklearn.metrics import precision_score, recall_score

print("The recall for the train set is ",recall_score(Y_train, tree_clf.predict (X_train)))
print("The recall for the test set tree is ",recall_score(Y_test, Y2_test_Tree_predict))
print("The precision for the train set tree is ",precision_score(Y_train, tree_clf.predict (X_train)))
print("The precision for the test tree is ",precision_score(Y_test, Y2_test_Tree_predict))


# In[57]:


Y_test_proba = tree_clf.predict_proba(X_test)
Y_test_proba[:,1]

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true = Y_test, probas_pred = Y_test_proba[:,1])
print(" Precision = ", precisions, "\n", "Recalls = ", recalls, "\n", "Thresholds = ", thresholds)


# ## Precision plot graph 

# In[58]:


def plot_precision_recall_vs_threshold (precisions, recalls, thresholds, color = "k", label = None):
    plt.plot (thresholds, precisions[:-1], color+"--", label="Precision "+label)
    plt.plot (thresholds, recalls[:-1], color+"-", label="Recall "+label)
    plt.xlabel("Threshold")
    plt.legend(loc="upper right")
    plt.ylim([0,1])


# In[59]:


plot_precision_recall_vs_threshold( precisions, recalls, thresholds, color = "b", label="Full tree")
plt.legend(loc = "best")


# ## Roc Curve Plotting

# In[60]:


from sklearn.metrics import roc_curve


# In[61]:


fpr, tpr, thresholds = roc_curve(Y_test, Y_test_proba[:,1])


# In[55]:


def plot_roc_curve (fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    
plot_roc_curve (fpr, tpr, "Full tree")
plt.show()


# In[64]:


from sklearn.metrics import roc_auc_score
print("AUC for random guess is:   ", 0.5 )
print("AUC for the Full tree is: ", roc_auc_score(Y_test, Y_test_proba[:,1]) )


# ## Random forest (RF)

# In[65]:


from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators = 1000, max_depth = 2, n_jobs= -1)
rnd_clf.fit (X_train, Y_train)
y_pred_rf = rnd_clf.predict(X_test)


# In[66]:


confusion_matrix(Y_test, y_pred_rf)


# In[67]:


recall_score(Y_test, y_pred_rf)


# In[68]:


precision_score(Y_test, y_pred_rf)


# In[69]:


y_rf_proba = rnd_clf.predict_proba(X_test)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test, y_rf_proba[:,1])
plot_roc_curve (fpr, tpr, "Full tree")
plot_roc_curve (fpr_rf, tpr_rf, "Random Forest")
plt.legend(loc = "best")
plt.show()


# In[70]:


roc_auc_score(Y_test, y_rf_proba[:,1])


# In[71]:


feature_score = rnd_clf.feature_importances_
feature_score


# In[72]:


feature_names = list(X_train)
df_feature_score = pd.DataFrame(data=feature_names, columns=["feature"])
df_feature_score["score"]= feature_score
df_feature_score = df_feature_score.sort_values(by=['score'], ascending=False)
df_feature_score.head()


# In[73]:


sns.barplot(y="feature", x="score", data=df_feature_score)


# ## GBM

# In[74]:


from sklearn import ensemble

clf_gb = ensemble.GradientBoostingClassifier()
clf_gb.fit(X_train, Y_train)


# In[75]:


y_pred_gb = clf_gb.predict(X_test)


# In[76]:


confusion_matrix(Y_test, y_pred_gb)


# In[77]:


recall_score(Y_test, y_pred_gb)


# In[78]:


precision_score(Y_test, y_pred_gb)


# In[79]:


y_gb_proba = clf_gb.predict_proba(X_test)
fpr_gb, tpr_gb, thresholds_gb = roc_curve(Y_test, y_gb_proba[:,1])

y_rf_proba = rnd_clf.predict_proba(X_test)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test, y_rf_proba[:,1])
plot_roc_curve (fpr, tpr, "Full tree")
plot_roc_curve (fpr_rf, tpr_rf, "Random Forest")
plot_roc_curve (fpr_gb, tpr_gb, "gradient boosting")
plt.legend(loc = "best")
plt.show()

