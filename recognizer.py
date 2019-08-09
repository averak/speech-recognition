# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Sequential
import numpy as np
import librosa, wave, glob, sys


class CNN(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.batch_size = 64
        self.classes = 11
        self.epochs = 10

        self.model_path = './ckpt/model.hdf5'

        # NNの構築
        self.build_NN()

    def train(self):
        ## -----*----- 学習 -----*----- ##
        # 特徴抽出
        self.extract_features()
        # 学習
        self.model.fit(self.datas['mfcc'], self.datas['label'],
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        # 学習モデルを保存
        self.model.save_weights(self.model_path)

    def build_NN(self):
        ## -----*----- NNの構築 -----*----- ##
        # モデルの定義
        self.model = Sequential([
            # Reshape((12, 13, 1), input_shape=(12, 13)),
            Conv2D(filters=11, kernel_size=(3, 3), input_shape=(12, 13, 1), strides=(1, 1),
                   padding='same', activation='relu'),
            Conv2D(filters=11, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(units=512, activation='relu'),
            Dropout(0.5),
            Dense(self.classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    def extract_features(self):
        ## -----*----- 学習データセットを用意 -----*----- ##
        # 教師データのロード
        files = glob.glob('./teacher_data/*/*.wav')
        self.datas = {'mfcc': [], 'label': []}
        cnt = 0
        for file in files:
            cnt += 1
            mfcc = self.to_mfcc(file)
            self.datas['mfcc'].append(mfcc)
            label = int(file.split('/')[2].split('_')[0])
            self.datas['label'].append(label)
            if cnt % 10 == 0:
                sys.stdout.write('\rmfcc_step:{0} / {1}'.format(cnt, len(files)))
                sys.stdout.flush()
            if cnt == 200:
                break
        print('\n')

        self.datas['mfcc'] = np.array(self.datas['mfcc'], dtype=np.float32)
        self.datas['label'] = np.array(self.datas['label'], dtype=np.uint8)

        # featuresの長さの連番数列を作る
        perm = np.arange(len(self.datas['mfcc']))
        # ランダムに並べ替え
        np.random.shuffle(perm)
        self.datas['mfcc'] = self.datas['mfcc'][perm]
        self.datas['label'] = self.datas['label'][perm]

    def to_mfcc(self, file, n_mfcc=12):
        ## -----*----- 音声データをMFCCに変換 -----*----- ##
        x, fs = librosa.load(file, sr=16000)
        mfcc = librosa.feature.mfcc(x, sr=fs, n_mfcc=n_mfcc)
        mfcc = np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1], 1))
        return np.array(mfcc, dtype=np.float32)

    def load_model(self):
        ## -----*----- 学習モデルの読み込み -----*----- ##
        self.model.load_weights(self.model_path)

    def predict(self, file):
        ## -----*----- 推論 -----*----- ##
        mfcc = self.to_mfcc(file)
        mfcc = np.expand_dims(mfcc, axis=0)
        score = self.model.predict(mfcc, batch_size=None, verbose=0)
        pred = np.argmax(score)
        return pred, score[0]


if __name__ == '__main__':
    cnn = CNN()
    # 学習
    # cnn.train()

    # 学習済みモデルの読み込み・推論
    cnn.load_model()
    print(cnn.predict(glob.glob('./shift_data/*.wav')[0]))
