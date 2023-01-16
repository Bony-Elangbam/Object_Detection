import cv2
import numpy as np

classNames = []
classFile = 'coco.names'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

choice = int(input("Enter your choice \n1.picture detection \n2.video detection\n"))

def picture():
    imgName = str(input("Enter the path of image : "))

    img = cv2.imread(imgName)

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 225, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(0)


def video():
    thres = 0.5
    nms_threshold = 0.1
    cap = cv2.VideoCapture(0)
    cap.set(3, 1288)
    cap.set(4, 720)
    cap.set(10, 150)

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 225, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)

if choice == 1:
    picture()
elif choice == 2:
    video()
else:
    print("Wrong choice !!!")


