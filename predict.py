import json
import os
import pickle
import tensorflow as tf
from utils import create_model, get_logger
from model import Model
from loader import input_from_line
from train import FLAGS, load_config


def main(_):
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)

    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = Model(config)

        logger.info("读取现有模型...")
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)  # 从模型路径获取ckpt
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            line = input("请输入测试句子:\n")
            result = model.evaluate_line(sess, input_from_line(line, FLAGS.max_seq_len,tag_to_id), id_to_tag)
            print(result)
            for item in result["entities"]:
                print('医疗实体提及:%s\t开始位置:%d\t结束位置:%d\t预定义类别:%s' % (item["word"], item["start_pos"], item["end_pos"], item["label_type"]))
            entities = list(set([item["word"] for item in result["entities"]]))
            print(entities)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)
