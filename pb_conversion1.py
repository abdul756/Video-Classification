# import tensorflow as tf
#
# pre_model = tf.keras.models.load_model("weight_new_all.hdf5")
# pre_model.save("saveddel")
# #
# from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from keras.preprocessing.image import  array_to_img
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# tf.keras.backend.clear_session()

loaded = tf.saved_model.load("saveddel")
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
folder_p = ['carcrash', 'falling', 'hitting', 'kicking', 'neutral', 'running', 'sitting', 'stealing', 'vandalizing',
            'walking']

print(folder_p[0])


import cv2

cap = cv2.VideoCapture(0)
max_height = 512
max_width = 512
#cap = cv2.VideoCapture('path to video')

while (True):
    ret, frame = cap.read()
    frame_v = cv2.resize(frame, (128, 128))
    # b, g, r = cv2.split(frame)  # get b,g,r
    # rgb_img = cv2.merge([r, g, b])  # switch it to rgb

    input_image = np.reshape(np.array(frame_v, dtype=np.float32), [1, 128, 128, 3]) / 255
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    out = infer(input_image)
    prediction=out['dense_1']
    print(folder_p[np.argmax(prediction)])

    # Window name in which image is displayed
    window_name = 'Activity detection'

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(frame, folder_p[np.argmax(prediction)], org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    # Displaying the image
    cv2.imshow(window_name, image)
    # plt.imshow(image)
    # plt.pause(0.2)
    # time.sleep(4)


#
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#     # When everything done, release the capture
 cap.release()
 cv2.destroyAllWindows()
