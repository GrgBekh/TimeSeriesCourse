import pandas as pd
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.model_selection import train_test_split
import joblib
SEED = 42

class ModelClass:

    def __init__(self, ndim = 15, data_pipline = None):

        self.ndim = ndim
        self.prophets = []
        self.cols = []
        self.catboost_models = [CatBoostRegressor(iterations=30,
                                                  learning_rate=0.05,
                                                  depth=4,
                                                  verbose=0) for _ in range(ndim)]
        self.categorical_features = []
        self.data_pipeline = data_pipline

    def fit_prophets(self, tsdf):
        self.prophets = [Prophet(holidays=tsdf['holidays']) for _ in range(self.ndim)]
        tsdf_cols = tsdf['sales'].columns
        self.cols = tsdf_cols
        for idx, y_col in enumerate(tsdf_cols):
            train_df = pd.DataFrame([tsdf['sales'].index, tsdf['sales'][y_col]]).T
            train_df.columns = ['ds', 'y']
            self.prophets[idx].fit(train_df)
        return self

    def predict_prophets(self, horizon = 30):
        model_predicts = []
        times = {}
        for idx, mdl in enumerate(self.prophets):
            future = mdl.make_future_dataframe(periods=horizon)
            forecast = mdl.predict(future)
            model_predicts.append(forecast['yhat'].clip(lower=0))
            times['ds'] = forecast['ds']

        list_of_cols = ['ds'] + [x + '_hat' for x in self.cols]
        preds = pd.concat([times['ds']] + model_predicts, axis= 1)
        preds.columns = list_of_cols
        return preds

    def fit_catboost(self, tsdf):
        y = tsdf['sales']
        prophet_predictions = self.predict_prophets(horizon=0).drop('ds', axis=1)
        calendar_features = tsdf['calendar'].drop(['wm_yr_wk', 'date_id'], axis=1).reset_index(drop=True)
        price_features = tsdf['prices'].reset_index(drop=True)

        # Combine features
        X = pd.concat([prophet_predictions, calendar_features, price_features], axis=1)

        # Identify categorical features
        self.categorical_features = calendar_features.select_dtypes(include=['category', 'object']).columns.tolist()

        for col in self.categorical_features:
            X[col] = X[col].astype('category')

        for idx, col in enumerate(self.cols):
            y_col = y[col].reset_index(drop=True)

            # Split the data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y_col, test_size=0.2, random_state=42)

            # Train CatBoost model
            self.catboost_models[idx].fit(X_train, y_train, eval_set=(X_val, y_val),
                                          cat_features=self.categorical_features)

    def predict_catboost(self, future_df, horizon):
        # Step 1: Generate Prophet predictions for the full range including the horizon
        prophet_full_predictions = self.predict_prophets(horizon=horizon)

        # Step 2: Extract only the horizon part
        prophet_predictions = prophet_full_predictions.tail(horizon).reset_index(drop=True)
        tsds = prophet_predictions['ds']
        prophet_predictions = prophet_predictions.drop('ds', axis=1)

        # Step 3: Process future calendar features
        calendar_features = future_df['calendar'].drop(['wm_yr_wk', 'date_id'], axis=1).reset_index(drop=True).iloc[:horizon]
        price_features = future_df['prices'].reset_index(drop=True).iloc[:horizon]

        # Combine features
        X_future = pd.concat([prophet_predictions, calendar_features, price_features], axis=1)

        # Convert categorical features
        for col in self.categorical_features:
            X_future[col] = X_future[col].astype('category')

        # Predict using CatBoost models
        catboost_predictions = []
        for idx, mdl in enumerate(self.catboost_models):
            preds = mdl.predict(X_future)
            catboost_predictions.append(preds)

        # Combine predictions into a DataFrame
        combined_predictions = pd.DataFrame({
            'ds': tsds
        })

        for idx, col in enumerate(self.cols):
            combined_predictions[col + '_hat'] = catboost_predictions[idx][:horizon]

        return combined_predictions

    def save_model(self, filepath):
        joblib.dump((
                     self.prophets,
                     self.cols,
                     self.catboost_models,
                     self.categorical_features,
                     self.data_pipeline
        ),
                     filepath)

    def load_model(self, filepath):
        self.prophets, self.cols, self.catboost_models, self.categorical_features, self.data_pipeline = joblib.load(filepath)
        return self

    def set_pipeline(self, pipe):
        self.data_pipeline = pipe

    @staticmethod
    def eval_preds(preds, true, prices):
        #По сути wrmse с весами ввиде цены, на дорогие товары дороже ошибаться
        return ((((preds.drop('ds', axis = 1) - true)**2) * prices).sum(axis = 0) / len(preds))**0.5

