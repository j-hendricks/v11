# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# from tensorflow.keras.utils import to_categorical
# from scipy.stats import linregress
# import sqlalchemy
# from sqlalchemy import create_engine, func
# from sqlalchemy.ext.automap import automap_base
# from sqlalchemy.orm import Session

# NOT UPLOADED TO GITHUB
# from config import PASSWORD, USERNAME, DATABASE_NAME, ENDPOINT


# engine = create_engine(f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{ENDPOINT}/{DATABASE_NAME}')
# Base = automap_base()
# Base.prepare(engine, reflect=True)
# Base.classes.keys()



# #Assigning tables to variables
# listings = Base.classes.listings
# calendar = Base.classes.calendar
# reviews = Base.classes.reviews

# session = Session(engine)
# listings_id = session.query(listings.id)

# sql_query_listings = pd.read_sql_query ('''
#                                SELECT
#                                *
#                                FROM listings_clean
#                                ''', engine)




df_list = pd.read_csv("Resources/cleandata/clean_listings.csv")

# df_list = pd.DataFrame(sql_query_listings, columns = sql_query_listings.keys())
# df_list = df_list.rename(columns={'id': 'listing_id'})
# df_list.head()

# sql_query_calendar = pd.read_sql_query ('''
#                                SELECT
#                                *
#                                FROM calendar_clean
#                                ''', engine)

# df_cal = pd.DataFrame(sql_query_calendar, columns = sql_query_calendar.keys())
# df_cal.head()

df_cal = pd.read_csv("Resources/cleandata/clean_calendar.csv")

df_cal_g = df_cal.groupby(['listing_id','date']).mean()

df_cal_g = df_cal_g.reset_index(level=['date'])

df = df_list.merge(df_cal_g, how='inner', on='listing_id')



df.rename(columns={"price_y":'price','date':'month'}, inplace=True)



df.drop(columns=['state','listing_id','price_x'], inplace=True)










df['bedrooms']=df['bedrooms'].apply(np.sqrt)


df['bathrooms'] = df['bathrooms'].apply(lambda x: pow(x,1/3))




df['accommodates']=df['accommodates'].apply(np.log10)




enc = OneHotEncoder(sparse=False)
encode_df = pd.DataFrame(enc.fit_transform(df[['city','zipcode','month']]))
encode_df.columns = enc.get_feature_names(['city','zipcode','month'])


df = df.merge(encode_df, left_index=True, right_index=True)
df.drop(columns=['city','zipcode','month'], inplace=True)


df.head()


X = df.drop(columns=['price'])
y = df['price']


print(X.shape)
print(y.shape)


X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# # Scale the data
# X_train_scaled = X_scaler.transform(X_train)
# X_test_scaled = X_scaler.transform(X_test)

# # %%
# X_train_scaled.shape

# # %%
# type(X_train_scaled)

# # %%
# X_train_scaled.shape

# # %%
# X_train_scaled

# # %%
# y_train

# # %%
# input_dim = X.shape[1]

# model = Sequential([
#     Dense(200, input_dim = input_dim, activation='relu'),
#     Dense(100, activation='relu'),
#     Dense(50, activation='relu'),
#     Dense(25, activation='relu'),
#     Dense(1)
# ])

# model.summary()

# # %%
# tf.keras.utils.plot_model(model)

# # %%
# model.compile(optimizer='adam',loss='mse')

# # %%
# history = model.fit(X_train_scaled, y_train, epochs=500, 
#                     validation_data=(X_test_scaled, y_test))

# # %%
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.plot(loss, label='loss')
# plt.plot(val_loss, label='val_loss')
# plt.legend()
# plt.show()

# # %%
# model.save('model.hdf5')
# y_pred = model.predict(X_test_scaled)

# # %%
# plt.scatter(y_pred, y_test)
# plt.ylabel('actual price')
# plt.xlabel('predicted price')

# # %%
# y_pred.tolist()

# # %%
# y_test.values.tolist()

# # %%
# results_df = pd.DataFrame({'predicted price':y_pred[:,0]})
# results_df.head()

# # %%
# results_df['actual price'] = y_test.values

# # %%
# results_df.head(20)

# # %%
# results_df['diff'] = abs(results_df['predicted price'] - results_df['actual price'])
# results_df

# # %%
# rmse = np.sqrt(history.history['val_loss'][-1])
# diffs = results_df['diff'].sum() / results_df.shape[0]
# results_df['lower'] = round(results_df['predicted price'] - 2*rmse)
# results_df['upper'] = round(results_df['predicted price'] + 2*rmse)

# results_df.loc[results_df['lower'] < 20, 'lower'] = 20

# results_df['accurate'] = (results_df['lower'] <= results_df['actual price']) & (results_df['actual price'] <= results_df['upper']) 
# results_df.tail()

# # %%
# results_df.head(20)

# # %%
# results_df[results_df['accurate'] == False]

# # %%
# results_df['accurate'].value_counts()

# # %%
# results_df['accurate'].value_counts()[1] / results_df['accurate'].value_counts().sum()

# # %%
# _, _, r_value, _, _ = linregress(results_df['predicted price'].values,results_df['actual price'].values)

# # %%
# print('r value:',r_value)

# # %%
# import pickle
# filename='model.pkl'
# pickle.dump(model, open(filename, 'wb'))

# # %%


# # %%


# # %%



