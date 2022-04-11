import os
from PIL import Image

from predict import predict

data_path = "../data/test/1/"
for i in range(0, 35):
    img_path = os.path.join(data_path, "%d.jpg" % i)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    num, prob = predict(img)
    print(img_path, num)
