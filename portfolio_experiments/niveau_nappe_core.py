import os 
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF

def add_date_features(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['weekday'] = df.index.weekday
    df['day'] = df.index.day

    return df

def add_derivate_features(df, col):
    df[f'diff_{col}'] = df[col].diff().fillna(0)
    df[f'diff30_{col}'] = df[col].diff(periods=30).fillna(0)
    df[f'diff90_{col}'] = df[col].diff(periods=90).fillna(0)
    df[f'diff180_{col}'] = df[col].diff(periods=180).fillna(0)

    df[f'diff_diff_{col}'] = df[f'diff_{col}'].diff().fillna(0)

    return df

def build_imputer_model():
    
    kernel = Matern()

    scaler = StandardScaler()

    regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)

    imputer = Pipeline([('sc', scaler), ('reg', regressor)])

    return imputer

def download_data(code_bss, fname):
    """Download the last 20000 measures
    """
    url = f'https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques.csv?code_bss={code_bss}&size=20000&sort=desc'
    if os.path.isfile(fname):
        data = pd.read_csv(fname, delimiter=";", index_col='date_mesure', parse_dates=['date_mesure'])
    else:
        data = pd.read_csv(url, delimiter=';', index_col='date_mesure', parse_dates=['date_mesure'])
        data.to_csv(fname, sep=';')
    return data.sort_index()

class RootMeanSquaredScaledError(tf.keras.metrics.Metric):
    def __init__(self, historical_mean_squared_difference, name='rmsse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hmsd = historical_mean_squared_difference
        self.rmsse = 0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.rmsse = tf.sqrt(tf.reduce_mean((y_true - y_pred)**2) / self.hmsd)
        
    def result(self):
        return self.rmsse
    
    def reset_state(self):
        self.rmsse = 0
        
def weightedAveragePercentageError(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae / tf.reduce_mean(y_true)

def compile_and_fit(model, window, patience=2, epochs=20, verbose=0):
    #hmsd = np.mean(window.train_df.niveau_nappe_eau.diff()[1:]**2)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    
    checkpoint_file = f'Checkpoints/{model.name}/best_weights'
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        monitor='val_loss',
        verbose=verbose,
        mode='min',
        save_best_only=True
    )
    
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
    
    history = model.fit(window.train, validation_data=window.val, epochs=epochs, callbacks=[early_stopping, checkpoint])
    
    model.load_weights(checkpoint_file)
    
    return history

def forecast(model, name, test_df, input_width, label_width, train_mean, train_std, stop_at='2022-01-15', label_index = None):
    
    test_df_tmp = test_df.copy()
        
    while test_df_tmp.index.max() < pd.Timestamp(stop_at):
                
        last_history = np.expand_dims(test_df_tmp.tail(input_width).values, axis=0)

        forecast = model.predict(last_history).squeeze()

        if label_index is not None:
            forecast = forecast[label_index]

    #     forecast = forecast * train_std + train_mean

    #     print("forecast")
    #     print(forecast)

        forecast_index = pd.date_range(start=test_df_tmp.index[-1], end=test_df_tmp.index[-1] + pd.Timedelta(label_width, unit='D'))[1:]

        df_pred = pd.DataFrame(data=forecast, index=pd.Index(forecast_index, name='DATE'), columns=['niveau_nappe_eau'])

        # Adding date features
        df_pred = add_date_features(df_pred)

        # adding derivate features
        df_pred = add_derivate_features(df_pred, 'niveau_nappe_eau')
        
        # normalize
        df_pred = (df_pred - train_mean) / train_std
        
        # cancel normalization on the the forecasted values
        df_pred['niveau_nappe_eau'] = forecast
        
        # Add the new data to end of the frame and remove the same amount of rows at the beginning.
        # This is just for not using to much memory
        test_df_tmp = test_df_tmp.append(df_pred)
    
    #     print("series")
    #     print(series)

    return_indices = pd.date_range(periods=label_width, end=pd.Timestamp(stop_at))
    
    return test_df_tmp.loc[return_indices]

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None, batch_size=64, shuffle=True):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.shuffle = shuffle
        
        self.batch_size = batch_size
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'History length: {self.input_width}',
            f'Horizon length: {self.label_width}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
        ])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
            
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
        
    def plot(self, model=None, plot_col='niveau_nappe_eau', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')
        plt.tight_layout()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data,
                                                                      targets=None,
                                                                      shuffle=self.shuffle,
                                                                      sequence_length=self.total_window_size,
                                                                      sequence_stride=1,
                                                                      sampling_rate=1,
                                                                      batch_size=self.batch_size
                                                                     )
        ds = ds.map(self.split_window)
        
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

class RNForecaster(tf.keras.Model):
    def __init__(self, kernel_size, filters, label_width, n_blocks=2, use_batch_norm=False,**kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.bn = use_batch_norm
        
        self.conv_init = tf.keras.layers.Conv1D(filters=filters, kernel_size=self.kernel_size, activation='relu', padding='same', name='conv_init')
        self.bn_init = tf.keras.layers.BatchNormalization(name='bn_init')
        
        self.convs = []
        self.bns = []
        self.adds = []
        
        for stage in range(n_blocks):
            conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=self.kernel_size, activation='relu', padding='same', name=f"conv_{stage}")
            self.convs.append(conv)
                
            bn = tf.keras.layers.BatchNormalization(name=f"bn_{stage}")
            self.bns.append(bn)
                
            self.adds.append(tf.keras.layers.Add(name=f'add_{stage}'))
                
                
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=label_width)
        self.reshape = tf.keras.layers.Reshape((label_width, -1))
        
    def call(self, inputs):
        out = self.conv_init(inputs)
        
        if self.bn:
            out = self.bn_init(out)
            
        for i in range(len(self.convs)):
            out_shortcut = out
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.adds[i]([out, out_shortcut])
        
        out = self.flatten(out)
        
        out = self.dropout(out)
        out = self.dense1(out)        
        out = self.dense2(out)
        out = self.reshape(out)
        
        return out
    
    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=(input_shape))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
