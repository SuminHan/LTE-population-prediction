import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import time, tqdm
import mymodels, utils
from utils import *
import warnings
import keras
from keras.callbacks import TensorBoard





def model_define(args, metadata):
    P, Q = args.P, args.Q
    data_width, data_height = metadata['data_width'], metadata['data_height']

    X  = layers.Input(shape=(P, data_height, data_width, 1), dtype=tf.float32)
    TE  = layers.Input(shape=(P+Q, 2), dtype=tf.int32)
    if args.model_name == 'MyARLSTM':
        tmp_model = mymodels.MyARLSTM(args, metadata)
        Y = tmp_model(X, TE)
        model = keras.models.Model((X, TE), Y)
        model_name = tmp_model.model_name

    if args.model_name == 'MyConvLSTM':
        tmp_model = mymodels.MyConvLSTM(args, metadata)
        Y = tmp_model(X, TE)
        model = keras.models.Model((X, TE), Y)
        model_name = tmp_model.model_name
        
    if args.model_name == 'MyRWRModel':
        tmp_model = mymodels.MyRWRModel(args, metadata)
        Y, _, _, _, _ = tmp_model(X, TE)
        model = keras.models.Model((X, TE), Y)
        model_name = tmp_model.model_name

    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--label', type=str, default=f'hongdae')
    parser.add_argument('--model_name', type=str, default=f'MyARLSTM')
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--max_data_size', type=int, default=24*7*10) # history sequence
    parser.add_argument('--teacher', type=float, default=0)
    parser.add_argument('--P', type=int, default=6) # history sequence
    parser.add_argument('--Q', type=int, default=1) # prediction sequence
    parser.add_argument('--S', type=int, default=1) # dataset step
    parser.add_argument('--K', type=int, default=1) # stack of layers
    parser.add_argument('--D', type=int, default=64) # stack of layers
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    parser.add_argument('--conv_kernel_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default=f'sgd')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--patience_stop', type=int, default=10)
    parser.add_argument('--patience_lr', type=int, default=3)
    
    args = parser.parse_args()
    print(args)

    dataset, metadata = utils.load_data(args)
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY = dataset
    print('trainX:', trainX.shape,  'trainTE:', trainTE.shape,  'trainY:', trainY.shape)
    print('valX:', valX.shape,      'valTE:', valTE.shape,      'valY:', valY.shape)
    print('testX:', testX.shape,    'testTE:', testTE.shape,    'testY:', testY.shape)

    # define the model
    model, model_name = model_define(args, metadata)
    model_checkpoint = f'./model_checkpoint/{args.label}/{model_name}'
    model_logs = f'./model_logs/{args.label}/{model_name}'

    if args.restore_model:
        try:
            model.load_weights(args.model_checkpoint)
            print('model restore successful')
        except:
            print('there exists no pretrained model')
    
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(args.learning_rate)

    model.compile(loss=custom_mae_loss, optimizer=optimizer)

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience_stop)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=args.patience_lr)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(model_checkpoint, save_weights_only=True, \
                    save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    time_callback = utils.TimeHistory()
    tb_callback = TensorBoard(log_dir=model_logs, histogram_freq=1, write_graph=True, write_images=True)

    # Suppress the error message
    warnings.filterwarnings("ignore", message="MutableGraphView::SortTopologically error: detected edge(s) creating cycle(s)")

    # Generate the model visualization (which may still contain the cycle)

    # Reset the warning filters (so other warning messages are still displayed)
    warnings.resetwarnings()
    
    model.fit((trainX, trainTE), trainY,
                batch_size=args.batch_size,
                epochs=args.max_epoch,
                verbose=1,
                validation_data=((valX, valTE), valY),
                callbacks=[early_stopping, model_ckpt, reduce_lr, tb_callback],
    )

    predY = model.predict((testX, testTE), batch_size=1) * 10000
    labelY = testY * 10000
    print(f'{args.model_name} test result:', metric(labelY, predY))
    
