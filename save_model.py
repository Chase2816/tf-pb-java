import tensorflow as tf
from core.yolov3 import YOLOV3
from tensorflow.saved_model import signature_def_utils, signature_constants, tag_constants
from tensorflow.saved_model import utils as save_model_utils
import os

wd = os.getcwd()
savemodel_file_path = wd + "/yolov3/1"

ckpt_file = r"E:\files\tensorflow-yolov3\checkpoint\yolov3_test_loss=2.5620.ckpt-25"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2",
                     "pred_multi_scale/concat"]

if not os.path.exists(savemodel_file_path):
    # os.mkdir(savemodel_file_path)
    os.makedirs(savemodel_file_path)

img_size = 416
num_channels = 3
with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, num_channels], name='input_data')

model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
print("{} trainable variables".format(len(tf.trainable_variables())))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    x_op = sess.graph.get_operation_by_name('input/input_data')
    x = x_op.outputs[0]
    pred_op = sess.graph.get_operation_by_name('pred_multi_scale/concat')
    pred = pred_op.outputs[0]

    print("prediction signature")
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={"input": save_model_utils.build_tensor_info(x)},
        outputs={"output": save_model_utils.build_tensor_info(pred)},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder(savemodel_file_path)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                         signature_def_map={"predict": prediction_signature})

    print("save...")
    builder.save()

sess.close()
