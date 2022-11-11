import cv2
from keras_cv_attention_models import yolor
from keras_cv_attention_models.coco import data

model = yolor.YOLOR_CSP(pretrained="coco")
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    status, frame = webcam.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    preds = model(model.preprocess_input(frame))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    
    if status:
        print(lables)
        print(
            [int(list(lables).count(0)),int(list(lables).count(2))]
            ) # 자동차 = 2, 사람은 0    출력은 [사람수, 차량수] 와 같은 형태로 출력됨.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
