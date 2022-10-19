#building damage Classification Prediction Model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

#reading the dataset to python 
df=pd.read_csv("csv_building_structure.csv") 

# Handling Missing values
df.isna().sum()
index=[83766, 131558, 131579, 131627, 131629, 131640, 131652, 131654,131655, 131656, 131929, 131932]
for i in index:
    df.drop(i,axis=0,inplace=True)
    
def impute_plinth_area(cols):
    rtype=cols[0]
    plinth=cols[1]
    if pd.isnull(plinth):
        if rtype == "Bamboo/Timber-Heavy roof" :
            return 360
        elif rtype == "Bamboo/Timber-Light roof" :
            return 345
        elif rtype == "RCC/RB/RBC" :
            return 700
    else:
        return plinth

df["roof_type"]=df["roof_type"].fillna(df["roof_type"].mode()[0])
df["plinth_area_sq_ft"]=df[["roof_type","plinth_area_sq_ft"]].apply(impute_plinth_area,axis=1)

#Removing the duplicates
df.drop_duplicates(keep='first',inplace=True)

#Handling of outlirers in the dataset
Q1=df['age_building'].quantile(0.25)
Q3=df['age_building'].quantile(0.75)
IQR=Q3-Q1    
LL=Q1-(1.5*IQR)
UL=Q3+(1.5*IQR)
df['age_building']= np.where(df['age_building']>UL,UL,np.where(df['age_building']<LL,LL,
                                                                           df['age_building']))
#Feature Reduction
Raw_Data=df.copy()
Raw_Data['plinth_area_sq_ft'] = np.log(Raw_Data['plinth_area_sq_ft'])
Raw_Data['height_ft_pre_eq'] = np.log(Raw_Data['height_ft_pre_eq'])
Raw_Data.drop(['building_id', 'district_id', 'vdcmun_id', 'ward_id'],axis=1,inplace=True)
Raw_Data.drop(['has_superstructure_timber',
       'plan_configuration', 'has_superstructure_cement_mortar_brick',
       'has_superstructure_bamboo', 'has_superstructure_stone_flag',
       'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_rc_non_engineered', 'has_superstructure_other',
       'has_superstructure_rc_engineered'],axis=1,inplace=True)

#Encoding the Target column
lbl=LabelEncoder()
Raw_Data["damage_grade"]=lbl.fit_transform(Raw_Data["damage_grade"])

X = Raw_Data.drop(['damage_grade'],axis=1)
y= Raw_Data['damage_grade']

#Encoding the Categorical Column
import category_encoders as ce 
# # Define catboost encoder
cbe_encoder = ce.cat_boost.CatBoostEncoder()
  
# # Fit encoder and transform the features
cbe_encoder.fit(X, y)
X = cbe_encoder.transform(X)

#Sampling of dataset
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
renn = RepeatedEditedNearestNeighbours()
X_res, y_res = renn.fit_resample(X, y)
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=34,stratify=y_res)

#Fitting the Model
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

# # Saving the model to disk
pickle.dump(classifier, open("model.pkl","wb"))
