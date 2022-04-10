import os
from PIL import Image

from predict import predict

img_path = "../test/902.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)
num, prob = predict(img)
print(num)
