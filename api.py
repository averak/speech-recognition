# -*- coding: utf-8 -*-
from flask import Flask, make_response, jsonify
import logging
from infer import Inference

# api（ログを無効化）
api = Flask(__name__)
logging.getLogger('werkzeug').disabled = True


@api.before_first_request
def startup():
    ## -----*----- サーバ起動後に実行 -----*----- ##
    global pred
    pred = Inference()


@api.route('/', methods=['POST'])
def predict():
    ## -----*----- 推論結果を返す -----*----- ##
    pattern = pred.inference()
    if pattern == 'Miss':
        color = 31
    else:
        color = 36
    pred.console.draw(*pred.string, '\033[90m推論中\033[0m', '\033[{0}m--{1}--\033[0m'.format(color, pattern),
                      *pred.meter)

    # 推論結果を返す
    result = {'result': pattern}
    return make_response(jsonify(result))


@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    api.run(host='localhost', port=3001, debug=False)
