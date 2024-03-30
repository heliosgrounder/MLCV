import cv2
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image


def demo(file_name):
    DEEPLAB_PALETTE = Image.open("helios_package/colorpalette.png").getpalette()

    deep_model = "helios_package/deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite"
    num_threads = 4
    # file_name = "test/picture.png"

    interpreter = tf.lite.Interpreter(model_path=deep_model, num_threads=num_threads)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]["index"]
    deeplabv3_predictions = interpreter.get_output_details()[0]["index"]

    color_image = cv2.imread(file_name)

    # Normalization
    prepimg_deep = cv2.resize(color_image, (256, 256))
    prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
    prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
    prepimg_deep = prepimg_deep.astype(np.float32)
    cv2.normalize(prepimg_deep, prepimg_deep, -1, 1, cv2.NORM_MINMAX)

    # Run model - DeeplabV3-plus
    interpreter.set_tensor(input_details, prepimg_deep)
    interpreter.invoke()

    # Get results
    predictions = interpreter.get_tensor(deeplabv3_predictions)[0]

    # Segmentation
    outputimg = np.uint8(predictions)
    # outputimg = cv2.resize(outputimg, (camera_width, camera_height))
    outputimg = cv2.resize(outputimg, (color_image.shape[1], color_image.shape[0]))
    outputimg = Image.fromarray(outputimg, mode="P")
    outputimg.putpalette(DEEPLAB_PALETTE)
    outputimg = outputimg.convert("RGB")
    outputimg = np.asarray(outputimg)
    outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)
    imdraw = cv2.addWeighted(color_image, 1.0, outputimg, 0.9, 0)

    # cv2.putText(imdraw, fps, (camera_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.putText(
        imdraw,
        "",
        (color_image.shape[1] - 170, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (38, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return imdraw


@pytest.mark.parametrize(
    "file_name,expected_shape",
    [
        ("test/src/test.png", (360, 480, 3)),
        ("test/src/test_shape_1.jpg", (360, 540, 3)),
        ("test/src/test_shape_2.jpg", (360, 640, 3)),
        ("test/src/test_shape_3.jpg", (360, 541, 3)),
    ],
)
def test_shape(file_name, expected_shape):
    assert demo(file_name).shape == expected_shape
