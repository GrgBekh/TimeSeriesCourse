import pandas as pd
from sklearn.experimental import enable_iterative_imputer #noqa
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin

class SalesToTimeSeriesFormat:

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, sales_df, store_name = 'STORE_1'):

        sales_cp = sales_df.copy().query("store_id == @store_name").drop('store_id', axis=1)
        sales_multidim = sales_cp.pivot_table(values='cnt', columns='item_id', index='date_id', aggfunc='sum')
        sales_multidim.reset_index(inplace=True)
        sales_multidim.set_index('date_id', inplace=True)
        sales_multidim.index.name = 'date_id'
        sales_multidim.sort_index(inplace=True)

        return sales_multidim


class PricesToTimeSeriesFormat:

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, prices_df, store_name='STORE_1'):

        prices_cp = prices_df.copy().query("store_id == @store_name")
        # Если вдруг окажется что есть две цены на один товар в одно время то возьмем среднюю
        prices_multidim = prices_cp.pivot_table(values='sell_price', index='wm_yr_wk', columns='item_id', aggfunc='mean')
        prices_multidim.reset_index(inplace=True)  # Ensure wm_yr_wk is a column
        prices_multidim.set_index('wm_yr_wk', inplace=True)  # Set wm_yr_wk as the index
        prices_multidim.index.name = 'wm_yr_wk'  # Set the index name
        prices_multidim.sort_index(inplace=True)

        return prices_multidim

class CalendarToTimeSeriesFormat:

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, calendar_df, store_name='STORE_1'):

        calendar_cp = calendar_df.copy()
        calendar_cp.index = pd.to_datetime(calendar_cp.index, format='%Y-%m-%d')
        calendar_cp.sort_index(inplace=True)
        cashback_column_name = 'CASHBACK_' + store_name
        calendar_cashback_col = calendar_cp[cashback_column_name]
        calendar_cp.drop(['CASHBACK_STORE_1', 'CASHBACK_STORE_2', 'CASHBACK_STORE_3', 'wday', 'month', 'year', 'weekday'],
                      axis=1,
                      inplace=True)

        fill_values = {'event_name_1': 'noevent',
                       'event_name_2': 'noevent',
                       'event_type_1': 'notype',
                       'event_type_2': 'notype'
        }
        calendar_cp.fillna(value = fill_values, inplace=True)

        calendar_cp[cashback_column_name] = calendar_cashback_col

        return calendar_cp


class MultiTableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sales_transformer = SalesToTimeSeriesFormat(),
                 prices_transformer = PricesToTimeSeriesFormat(),
                 calendar_transformer = CalendarToTimeSeriesFormat(),
                 store_name='STORE_1'):
        self.sales_transformer = sales_transformer
        self.prices_transformer = prices_transformer
        self.calendar_transformer = calendar_transformer
        self.store_name = store_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Transform each table separately
        sales_transformed = self.sales_transformer.transform(X['sales'], self.store_name)
        prices_transformed = self.prices_transformer.transform(X['prices'], self.store_name)
        calendar_transformed = self.calendar_transformer.transform(X['calendar'], self.store_name)

        sales_transformed.columns = ['sales_' +  x.lstrip(self.store_name).lstrip('_') for x in
                                     sales_transformed.columns]
        prices_transformed.columns = ['price_' + x.lstrip(self.store_name).lstrip('_') for x in
                                     prices_transformed.columns]

        # Combine or return the transformed tables as needed
        return {
            'sales': sales_transformed,
            'prices': prices_transformed,
            'calendar': calendar_transformed
        }

class SelectiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer = IterativeImputer()):
        self.imputer = imputer

    def fit(self, X, y=None):
        self.imputer.fit(X['prices'])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        cols = X_transformed['prices'].columns
        idxs = X_transformed['prices'].index
        X_transformed['prices'] = pd.DataFrame(self.imputer.transform(X['prices']), columns=cols, index=idxs)
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class Joiner(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X, y=None):

        X_transformed = X.copy()

        calendar_cols = X_transformed['calendar'].reset_index().columns
        prices_cols = X_transformed['prices'].columns

        external_features = pd.merge(X_transformed['calendar'].reset_index(),
                                     X_transformed['prices'],
                                     left_on='wm_yr_wk',
                                     right_on=X['prices'].index,
                                     how='left').sort_index()


        X_transformed['calendar'] = external_features[calendar_cols].set_index('date')
        X_transformed['prices'] = external_features[prices_cols]
        X_transformed['prices'].index = X_transformed['calendar'].index
        X_transformed['sales'].index = X_transformed['calendar'].index

        return X_transformed

class HolydayExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X = None, y = None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        holidays = []

        # Iterate through each date in the calendar
        for date, row in X['calendar'].iterrows():
            event_1 = row['event_name_1']
            event_2 = row['event_name_2']

            # Check for events in both columns
            if event_1 != 'noevent':
                holidays.append({
                    'holiday': event_1,
                    'ds': date,
                    'lower_window': -3,
                    'upper_window': 3
                })
            if event_2 != 'noevent':
                holidays.append({
                    'holiday': event_2,
                    'ds': date,
                    'lower_window': -3,
                    'upper_window': 3
                })
            if event_1 != 'noevent' and event_2 != 'noevent' and date in X['calendar'].index:
                holidays[-2]['lower_window'] = -5
                holidays[-2]['upper_window'] = 5
                holidays.pop()

        X_transformed['holidays'] = pd.DataFrame(holidays)
        return X_transformed


