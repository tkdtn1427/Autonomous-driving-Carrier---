# 필요한 모듈 다운
import RPi.GPIO as GPIO                           
from time import sleep
import os
import argparse                                                                                     
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import motor



#비디오스트림 담당 클래스
class VideoStream:                                  
    """Camera object that controls video streaming from the Picamera"""
    #비디오스트림 초기설정
    def __init__(self, resolution=(1280, 720), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # 스트림의 첫 프레임 읽기
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    # 스레드  시작
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    #비디오 업데이팅
    def update(self):
        while True:
            # 카메라가 멈추면 스레드 멈추기
            if self.stopped:
                # 카메라 리소스 닫기
                self.stream.release()
                return

            # 그렇지 않으면 카메라 스트림 계속 grab
            (self.grabbed, self.frame) = self.stream.read()

    # 가장 최근 프레임 읽기
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# 입력인자 받기
parser = argparse.ArgumentParser()              
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

# 입력인자 관련 설정들
args = parser.parse_args()                     
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# tflite_runtime유무에 따라 사용할 모듈 결정 및 받기
pkg = importlib.util.find_spec('tflite_runtime')                                
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# edgetpu 사용유무에 따라 파일이름 바꾸기
if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

frame_count=0

# 현재 working derectory 위치 저장
CWD_PATH = os.getcwd()

# 모델 위치 저장
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# 라벨맵 위치 저장
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# 라벨맵 로드
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# tensorflow lite 모델 다운
# TPU 사용 시 학습한 양자화 model을 interpreter로 로드하여 interpreter 변수에 담기
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

# load된 interpreter에 tensor생성
interpreter.allocate_tensors()

# 모델 세부내용 저장
# 입력층에 해당하는 tensor의 정보와 출력에 해당하는 tensor의 정보를 details 변수에 담기
input_details = interpreter.get_input_details()                        
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# 비디오 프레임 계산 준비
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# 비디오스트림 시작
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# 비디오스트림 돌아갈 때 할 것들
while True:                                                   
    state = False
    if frame_count == 600:
        frame_count=0
    frame_count = frame_count+1

    # fps계산용 시간 세기
    t1 = cv2.getTickCount()

    # 프레임 읽기
    frame1 = videostream.read()

    # 해상도 조정
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # floating model을 쓸 경우 픽셀값 정규화
    if floating_model:                                                                  
        input_data = (np.float32(input_data) - input_mean) / input_std

    # 영상 frame을 입력으로 주어 input_details 변수를 세팅하고 작동
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 박스정보 저장
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    # 감지된 객체 라벨맵에서의 클래스 저장
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  
    # 감지된 객체 수 저장
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  

    # 객체 중심에 따른 우회전 좌회전할 라인을 화면에 그려놓음
    cv2.line(frame,(340,0),(340,720),(0,0,255),3)                                       
    cv2.line(frame,(930,0),(930,720),(0,0,255),3)

    # 감지된 객체 전부 박스칠
    for i in range(len(scores)):                                                      
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            # 박스의 각 x좌표,y좌표 및 중앙 좌표 계산
            ymin = int(max(1, (boxes[i][0] * imH)))                                    
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cent_x = (xmax+xmin)//2
            cent_y = (ymax+ymin)//2

            #박스칠
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)         
            
            # 라벨 그리기
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            # 너무 위에 그리기 않기
            label_ymin = max(ymin, labelSize[1] + 10)
            
            # 중앙좌표에 빨간 점 찍기
            cv2.circle(frame,(cent_x,cent_y),5,(0,0,255),-1)                            
           
            # 모터 움직이는 코드
             if object_name == 'Object':                                     
                state = True

                # 객체가 중앙일 경우 
                if(cent_x > 340 and cent_x <= 940):
                    # 일정크기 이하면
                    if(ymax-ymin < 430):
                        # 전진
                        motor.forward_r()
                    else:
                        #아니면 정지
                        motor.stop()
                
                # 객체가 너무 오른쪽이면
                elif(cent_x > 940):
                    #우회전
                    motor.right()
                # 둘다 아니면
                else:
                    motor.left()
    # 객체가 없으면       
    if state == False:
        motor.stop()
    
                
    # fps 찍기
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),           
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # 프레임에 저장해놓은 것들 보이기
    cv2.imshow('Object detector', frame)

    # 프레임 계산하기
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # q누르면 탈출
    if cv2.waitKey(1) == ord('q'):                                        
        break

# 마무리
cv2.destroyAllWindows()
videostream.stop()
