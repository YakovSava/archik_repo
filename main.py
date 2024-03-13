from os import listdir
from yolov5 import load

model = load("./best.pt")

model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

for file in listdir('./test/'):
    result = model('./test/'+file)
    result.save(save_dir='./test_result/exp')