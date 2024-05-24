# Guide-rknn

## Install dependencies on rockchip platform (in my case orange pi5 plus)

Basic pakages:

```
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip gcc
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-numpy
sudo apt-get install git
sudo apt-get install wget
sudo apt-get install python3-setuptools
```

**FOR 3588**

[version rknn-toolkit] (https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages)


```
#choose own version

wget https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so

sudo mv librknnrt.so /usr/lib/librknnrt.so

# choose correct version rknn-toolkit (look on your python3 version , in my case 3.10)
python3 --version

wget https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl

pip3 install rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl

sudo ln -s librknnrt.so librknn_api.so

```

**To test it:**

Download repsitory

`https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/examples/resnet18`

and run `test.py`

## Inference yolo-model
```
"""
Usage example 
"""
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from time import time

class Coco:
    CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
    

class Yolov8_rknn:
    def __init__(self,model:str,classes=Coco.CLASSES,imgsz=(640,640),obj_threshold = 0.5,nms_threshold = 0.65 ) -> None:
        self.model = None
        #init model
        self.model = self.__load_rknn(model)
        self.frame = None
        self.w = imgsz[0]
        self.h = imgsz[1]
        self.obj_threshold = obj_threshold
        self.nms_threshold  = nms_threshold


    def __load_rknn(self,link_model:str):
        """Load engine rknn file
        """ 
        print("Init RKNNLite --->")
        self.model = RKNNLite()
        print("Init yolov8 model --->")
        ret = self.model.load_rknn(link_model)
        ret = self.model.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)

        return self.model

    def filter_boxes(self,boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.obj_threshold)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self,boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def dfl(self, position):
        x = np.array(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)

        max_values = np.max(y, axis=2, keepdims=True)
        exp_values = np.exp(y - max_values)
        y = exp_values / np.sum(exp_values, axis=2, keepdims=True)

        acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        y = np.sum(y * acc_matrix, axis=2)

        return y


    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.h//grid_h, self.w//grid_w]).reshape(1,2,1,1)

        position = self.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy
    
    def post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores
    
    def __call__(self,img:np.ndarray):
        outputs = self.model.inference(inputs=[img])
        boxes, classes, scores = self.post_process(outputs)
        if boxes is not None:
            for box in boxes:
                box = box.astype(int)
                img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
            return img

        else:
            print("Not found box")
            return img
    def realse(self):
        self.model.release()

if __name__ == "__main__":
    model = Yolov8_rknn(model='yolov8.rknn')
    cap = cv2.VideoCapture(0)
    time_start = 0

    while True:
        ret,img = cap.read()
        img = cv2.resize(img,(640,640))
        img = model(img)
        cv2.imshow("img",img)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        fps = 1/(time() - time_start)
        print(fps)
        time_start = time()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    model.realse()

```



