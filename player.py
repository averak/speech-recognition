# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import wave, pyaudio


class PlayAudio(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.PATH = './response_voice/'
        self.mChunk = 1024

    def play(self, pattern):
        ## -----*----- patternと一致する音楽再生 -----*----- ##
        pa = pyaudio.PyAudio()
        wavFile = wave.open(self.PATH + pattern + '.wav', 'rb')
        stream = pa.open(
            format=pa.get_format_from_width(wavFile.getsampwidth()),
            channels=wavFile.getnchannels(),
            rate=wavFile.getframerate(),
            output=True,
        )
        voiceData = wavFile.readframes(self.mChunk)
        while len(voiceData) > 0:
            stream.write(voiceData)
            voiceData = wavFile.readframes(self.mChunk)
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == '__main__':
    play = PlayAudio()
    play.play('TV_on')
