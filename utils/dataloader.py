import os, tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

import datetime
from datetime import datetime, timedelta 


ORIGINAL_CRS = '+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 +x_0=400000 +y_0=600000 +ellps=bessel +towgs84=-115.8,474.99,674.11,1.16,-2.31,-1.63,6.43 +units=m +no_defs'



def point2cell(px, py, size=50):
    square = Polygon([(px - size/2, py - size/2), 
                      (px + size/2, py - size/2), 
                      (px + size/2, py + size/2), 
                      (px - size/2, py + size/2)])
    return square


def label2numpy(label):
    lte_fname = f'dataset/{label}_lte_2d_timeseries.npy'
    gdf_fname = f'dataset/{label}_lte_cell.geojson'

    rvals = np.load(lte_fname)
    new_gdf = gpd.read_file(gdf_fname)
    new_gdf.crs = ORIGINAL_CRS
    
    return new_gdf, rvals


def seq2instance(data, P, Q, S):
    num_step = data.shape[0]
    data_type = data.dtype
    num_sample = (num_step - P - Q + 1)//S
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(0, num_sample):
        j = i*S
        x[i] = data[j : j + P].astype(data_type)
        y[i] = data[j + P : j + P + Q].astype(data_type)
    return x, y

    
def set_all_dataset(data, timestamps, P=24, Q=24, S=6, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1
    
    TE = np.stack([timestamps.weekday, timestamps.hour], -1)

    train_split = train_ratio
    val_split   = train_ratio + val_ratio
    
    datasize = len(data)
    data = np.expand_dims(data, -1).astype(np.float32)
    
    train    = data[:int(datasize*train_split)]
    train_te = TE[:int(datasize*train_split)]
    val      = data[int(datasize*train_split):int(datasize*val_split)]
    val_te   = TE[int(datasize*train_split):int(datasize*val_split)]
    test     = data[int(datasize*val_split):]
    test_te  = TE[int(datasize*val_split):]
    
    trainX, trainY = seq2instance(train, P, Q, S)
    valX, valY = seq2instance(val, P, Q, S)
    testX, testY = seq2instance(test, P, Q, S)
    
    trainTE = seq2instance(train_te, P, Q, S)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val_te, P, Q, S)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test_te, P, Q, S)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY


def load_data(args):
    data_df = pd.read_csv('dataset/TOTAL_LTE_TABLE.csv', index_col='timestamp')
    cell_gdf = gpd.read_file('dataset/LTE_CELL_ID_NEW_BUILDING.geojson', driver='GeoJSON')
    cell_gdf.crs = ORIGINAL_CRS

    tgdf, tdata = label2numpy(args.label)
    tdata = tdata[:args.max_data_size] / 10000

    metadata = dict(args=args)
    data_width = tgdf['i'].max()+1 ; metadata['data_width'] = data_width
    data_height = tgdf['j'].max()+1; metadata['data_height'] = data_height
    area_building = tgdf['building'].values.reshape(data_height, data_width)
    metadata['area_building'] = area_building
    timestamp = pd.to_datetime(data_df.index)[:args.max_data_size]; metadata['timestamp'] = timestamp


    dataset = set_all_dataset(tdata, timestamp, 
                            P=args.P, Q=args.Q, S=args.S, 
                            train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    return dataset, metadata





