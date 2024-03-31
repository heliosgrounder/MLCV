import cv2
import numpy as np
# import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import streamlit as st

uploaded_file = st.file_uploader("Choose a image file (png)", type="png")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    color_image = cv2.imdecode(file_bytes, 1)

    DEEPLAB_PALETTE = Image.open("helios_package/colorpalette.png").getpalette()

    deep_model = "helios_package/deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite"
    num_threads = 4
    # file_name = "test/picture.png"

    # interpreter = tf.lite.Interpreter(model_path=deep_model, num_threads=num_threads)
    interpreter = Interpreter(model_path=deep_model, num_threads=num_threads)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]["index"]
    deeplabv3_predictions = interpreter.get_output_details()[0]["index"]


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

    # Now do something with the image! For example, let's display it:
    st.image(imdraw, channels="BGR")