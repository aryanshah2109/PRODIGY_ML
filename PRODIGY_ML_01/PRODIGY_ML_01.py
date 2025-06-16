# Dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.metrics import r2_score
import pickle
import os
from src.feature_engineering import FeatureEngineering

def evaluate(model,X_val,y_test):
    y_pred = model.predict(X_val)
    r2 = r2_score(y_test,y_pred)
    return r2

# reading data
base = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base, 'data', 'train.csv')
test_path = os.path.join(base, 'data', 'test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y = train_df['SalePrice']
X = train_df.drop(columns=['Id', 'SalePrice', 'Alley', 'PoolQC', 'Fence', 'Utilities', 'MiscFeature', 'MiscVal'])
X_test = test_df.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'Utilities', 'MiscFeature', 'MiscVal'])


# splitting using train test split
X_train, X_val, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# categories
numerical_features = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'TotalSF', 'HouseAge'
]


ordinal_features = [
    'LotShape', 'LandSlope', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
    'KitchenQual', 'Functional', 'FireplaceQu',
    'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive'
]
ordinal_categories = [
      ['Reg', 'IR1', 'IR2', 'IR3'],
      ['Gtl', 'Mod', 'Sev'],
      ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['NA', 'No', 'Mn', 'Av', 'Gd'],
      ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
      ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
      ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
      ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['NA', 'Unf', 'RFn', 'Fin'],
      ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
      ['N', 'P', 'Y']
]
nominal_features = [
      'MSZoning', 'Street', 'LandContour', 'LotConfig',
      'Neighborhood', 'Condition1', 'Condition2',
      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
      'Exterior1st', 'Exterior2nd', 'MasVnrType',
      'Foundation', 'Heating', 'CentralAir',
      'Electrical', 'GarageType', 'SaleType', 'SaleCondition'
]

# numerical pipeline
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='constant',fill_value='NA')

numerical_pipeline = Pipeline([
    ('num',numerical_imputer),
    ('scaler',StandardScaler())
])

# categorical pipelines
ordinal_pipeline = Pipeline([
    ('ord',categorical_imputer),
    ('ordencode',OrdinalEncoder(categories=ordinal_categories,handle_unknown='use_encoded_value',unknown_value=-1))
])

nominal_pipeline = Pipeline([
    ('nom',categorical_imputer),
    ('nomencode',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

# preprocessing pipeline
preprocessor_pipeline = ColumnTransformer([
    ('numerical',numerical_pipeline,numerical_features),
    ('ordinal',ordinal_pipeline,ordinal_features),
    ('nominal',nominal_pipeline,nominal_features)
])

# model pipeline
model_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('preprocess',preprocessor_pipeline),
    ('model',Ridge())
])

params={'model__alpha':[0.01,0.05,0.1,0.5,1,10,100,200,300]}
grid = GridSearchCV(model_pipeline,param_grid=params,scoring='neg_root_mean_squared_error',cv=5)
grid.fit(X_train,y_train)

# saving the model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, 'house_price_pipeline.pkl'), 'wb') as f:
    pickle.dump(grid, f)


