# -*- coding: utf-8 -*-
import os, sys, time, wave, glob, librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from collections import Counter
from recognizer import CNN
from console import Console


class Inference(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        # CNN構築
        self.clf = CNN()
        self.clf.load_model()
        # ラベル
        self.name = ['電気', 'エアコン', 'カーテン', 'テレビ', '扇風機', 'つけて', '消して', '開けて', '閉めて', '無音', 'その他']
        # PATH指定
        self.shift_format_path = './files/shift_format.wav'
        self.audio_path = './files/source.wav'
        self.shift_path = './shift_data'
        # コンソール出力
        self.console = Console('./files/design_infer.txt')

        # 教師データからフォーマットを取得
        self.read_format(self.shift_format_path)

    def read_format(self, path):
        ## -----*----- 教師データからフォーマットを取得 -----*----- ##
        wf = wave.open(path, 'rb')
        self.format = {'channel': wf.getnchannels(), 'width': wf.getsampwidth(),
                       'rate': wf.getframerate(), 'point': len(wf.readframes(wf.getnframes()))}
        wf.close()

    def inference(self):
        ## -----*----- 音声ファイルを推論 -----*----- ##
        self.meter = [''] * 11
        rate = {'object': [], 'command': []}
        self.audio_path = self.split_by_silence()
        dats, dats_frames = self.get_dats(self.audio_path)
        total_shift_size, shift_size = self.get_shift_size(dats_frames)
        self.save_shift_audio(dats, total_shift_size, shift_size)
        shift_audio_paths = ['{0}/shift_{1}.wav'.format(self.shift_path, n) for n in
                             range(int(len(glob.glob('{0}/*.wav'.format(self.shift_path)))))]
        for path in shift_audio_paths:
            pred, proba = self.clf.predict(path)
            index = int(path.rsplit('/', 1)[1].split('_')[1].split('.')[0])
            if (pred < 5) and (proba[pred] > 0.3) and (
                    index < int(len(glob.glob('{0}/*.wav'.format(self.shift_path))) * 0.5)):
                rate['object'].append(pred)
                self.meter[pred] += '■■■'
            elif (pred >= 5) and (pred < 9) and (proba[pred] > 0.3) and (
                    index > int(len(glob.glob('{0}/*.wav'.format(self.shift_path))) * 0.5)):
                rate['command'].append(pred)
                self.meter[pred] += '■■■'
            elif (pred >= 9):
                rate['command'].append(pred)
                self.meter[pred] += '■■■'
            self.string = [str(index), self.name[pred], str(int(proba[pred] * 100))]
            self.console.draw(*self.string, '\033[32m推論中\033[0m', '', *self.meter)
        rate['object'] = self.unique(rate['object'])
        rate['command'] = self.unique(rate['command'])

        try:
            object_status = Counter(rate['object']).most_common()[0][0]
            command_status = Counter(rate['command']).most_common()[0][0]
            if len((list(set([i for i in rate['command'] if i != 10 and i != 9])))) == 1:
                command_status = (list(set([i for i in rate['command'] if i != 10 and i != 9])))[0]
            pattern = self.to_sentence(object_status, command_status)
            return pattern
        except:
            return 'Miss'

    def split_by_silence(self):
        ## -----*----- 無音区間をカット -----*----- ##
        self.audio_path = './files/source.wav'
        sound = AudioSegment.from_file(self.audio_path, format='wav')
        chunks = split_on_silence(
            sound,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=150
        )
        for i, chunk in enumerate(chunks):
            chunk.export('./files/split_' + str(i) + '.wav', format='wav')
        if len(chunks) != 0:
            self.audio_path = './files/split_0.wav'
        return self.audio_path

    def to_sentence(self, object, command):
        ## -----*----- 推論結果を命令文に変換 -----*----- ##
        sentences = {'05': 'light_on', '06': 'light_off',
                     '15': 'air_on', '16': 'air_off',
                     '27': 'curtain_on', '28': 'curtain_off',
                     '35': 'TV_on', '36': 'TV_off',
                     '45': 'fan_on', '46': 'fan_off'}
        sentence = sentences.get(str(object) + str(command))
        if sentence == None:
            sentence = 'Miss'
        return sentence

    def get_dats(self, path):
        ## -----*----- フレーム数の取得 -----*----- ##
        dats = []
        wf = wave.open(path)
        wf_point = wf.readframes(wf.getnframes())
        dats_frames = len(wf_point)
        dats.append(wf_point)
        wf.close()
        return dats, dats_frames

    def get_shift_size(self, dats_size):
        ## -----*----- シフトする個数の取得 -----*----- ##
        total_shift_size = dats_size - self.format['point']
        shift_size = int(self.format['point'] * 0.1)
        while (total_shift_size % shift_size != 0) and (shift_size < int(self.format['point'] * 0.2)):
            shift_size += 1
        if shift_size == int(self.format['point'] * 0.2):
            shift_size = int(self.format['point'] * 0.1)
            total_shift_size -= 1
            while (total_shift_size % shift_size != 0) and (shift_size < int(self.format['point'] * 0.5)):
                shift_size += 1
        return total_shift_size, shift_size

    def save_shift_audio(self, dats, total_shift_size, shift_size):
        ## -----*----- シフトしたファイルを保存 -----*----- ##
        cnt = 0
        total_shift = 0
        self.remove_shif()
        while total_shift <= total_shift_size:
            wf = wave.open('{0}/shift_{1}.wav'.format(self.shift_path, cnt), 'wb')
            wf.setnchannels(self.format['channel'])
            wf.setsampwidth(self.format['width'])
            wf.setframerate(self.format['rate'])
            wf.writeframes(dats[0][int(shift_size) * cnt:self.format['point'] + int(shift_size) * cnt])
            wf.close()
            total_shift += shift_size
            cnt += 1

    def remove_shif(self):
        ## -----*----- shift_path内のファイルを削除 -----*----- ##
        target_paths = glob.glob('{0}/*.wav'.format(self.shift_path))
        for path in target_paths:
            os.remove(path)

    def unique(self, array):
        ## -----*----- 連続した要素を一つにまとめる -----*----- ##
        if len(array) == 0:
            return []
        ret = [array[0]]
        if len(array) > 1:
            for i in range(1, len(array)):
                if not (array[i] == 10 and ret[-1] == 10):
                    if not (array[i] == 9 and ret[-1] == 9):
                        ret.append(array[i])
        return ret


if __name__ == '__main__':
    pred = Inference()
    pred.inference()
    print (pred.string)
