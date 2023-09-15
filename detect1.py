# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from threading import Condition
from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
    SeekFrame,
)

########################################

# 이 클래스는 화면에 이미지를 랜더링하는 데 필요한 카메라 및 이미지 데이터를 포함한다,

class Renderer:
    """Contains camera and image data required to render images to the screen."""

    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True

# 새 프레임이 사용 가능할 떄마다 실행되는 비동기 콜백 함수이다.
def on_frame(_camera, camera_frame, renderer):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------	
    _camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    renderer: Renderer
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the renderer object.
    """

    with renderer.frame_condition:

        renderer.frame = camera_frame.thermography_float

        renderer.frame_condition.notify()



def on_event(camera, event_type, event_status, renderer):
    
    # 카메라가 어떤 상태인지와 카메라 시리얼 넘버? 출력
    print("{}: {}".format(str(event_type), camera.chipid))

    # 만약 camera가 연결이 되어있으면
    if event_type == SeekCameraManagerEvent.CONNECT:

        # renderer.busy : 랜더러가 현재 사용중인지 여부를 나타내는 플래그
        if renderer.busy:
            return

        # Claim the renderer.
        # This is required in case of multiple cameras.
        renderer.busy = True

        # renderer.camera : 현재 사용중인 카메라 객체의 참조
        renderer.camera = camera

        # Indicate the first frame has not come in yet.
        # This is required to properly resize the rendering window.
        # renderer.first_frame : 첫 번쨰 프레임을 받았는지 여부를 나타내는 플래그
        renderer.first_frame = True

        # Set a custom color palette.
        # Other options can set in a similar fashion.
        # camera.color_palette를 설정하여 열화상 이미지의 색상 팔레트를 설정할 수 있따.
        # 이 코드에서는 TYRIAN 팔레트로 설정되어 있다.
        camera.color_palette = SeekCameraColorPalette.TYRIAN

        # Start imaging and provide a custom callback to be called
        # every time a new frame is received.
        # Seek Thermal 카메라에서 새 프레임이 사용 가능할 때마다 실행할 콜백 함수를 등록하는 역할
        camera.register_frame_available_callback(on_frame, renderer) # 함수를 호출하여 새 프레임을 처리할 콜백 함수를 설정

        # 카메라에서 이미지 캡처 세션을 시작하고 캡처한 이미지 형식을 설정
        # SeekCameraFrameFormat.COLOR_ARGB888 형식은 열화상 이미지를 컬러 ARGB888 형식으로 캡처하도록 지시
        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

    # 카메라가 연결 해제되었을 떄 실행된다.
    # 연결 해제 이벤트가 발생하면 이미지 수집을 중단하고 렌더러 상태를 재설정
    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        # Check that the camera disconnecting is one actually associated with
        # the renderer. This is required in case of multiple cameras.
        if renderer.camera == camera:
            # Stop imaging and reset all the renderer state.
            camera.capture_session_stop()
            renderer.camera = None
            renderer.frame = None
            renderer.busy = False

    # 카메라 연결 또는 작동 중에 오류가 발생한 경우에 실행된다. 오류메시지 출력
    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    # 연결된 카메라가 페어링 준비가 되었을 때 실행
    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return

def apply_colormap(frame):
    frame = renderer.frame.data
    # 열화상 이미지의 최소 및 최대 온도 범위 설정 (예: 0°C에서 100°C)
    min_temperature = -10.0
    max_temperature = 40.0

    # 열화상 이미지를 8비트로 변환 (0-255 범위)
    normalized_frame = np.interp(frame, (min_temperature, max_temperature), (0, 255))
    normalized_frame = normalized_frame.astype(np.uint8)
    # normalized => 새로 150 가로 200


    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced_frame = clahe.apply(normalized_frame)

    # cv2.COLORMAP_JET 또는 다른 컬러 맵을 선택하여 적용
    # colormap = cv2.COLORMAP_WINTER
    # colored_frame = cv2.applyColorMap(enhanced_frame, colormap)


    return normalized_frame
###################################################
renderer = Renderer()
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    device = select_device(device)
    model = DetectMultiBackend('best.pt', device=device, dnn=dnn, data=data, fp16=half)
    source = str(source)
    manager = SeekCameraManager(SeekCameraIOType.USB)
    manager.register_event_callback(on_event, renderer)
    while True:

# 무한 루프 => 이 루프는 프레임을 지속적으로 수집하고 렌더링 한다.
        # Load model


        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
# with renderer.frame_condition : 블록 내에서 새로운 프레임을 기다리며
        with renderer.frame_condition:
            if renderer.frame_condition.wait(1000 / 1000.0):
                img = renderer.frame.data
                colored_img = apply_colormap(img)
                if len(colored_img.shape) == 3:
                    (height, width, _) = colored_img.shape
                else:
                    (height, width) = colored_img.shape
                #########################
                colored_img= np.repeat(colored_img[:, :, np.newaxis], 3, axis=2)
                t2 = colored_img
                im00=t2
                s = f'image {0}/{0} {source}: '
                im1 = cv2.resize(im00, (640,640), interpolation=cv2.INTER_LINEAR)
                im1 = letterbox(im00, imgsz, stride=stride, auto=pt)[0]  # padded resize
                
                im1 = im1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im1 = np.ascontiguousarray(im1)  # contiguous

                dataset=[[source,im1,im00,1,s]]
      
###################################3

    # if renderer.first_frame:
    #    renderer.first_frame = False


        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

        # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            try:
                x1,x2,y1,y2 = float(pred[0][0][0]), float(pred[0][0][1]), float(pred[0][0][2]), float(pred[0][0][3])
                print(x1,x2,y1,y2)
            except : 
                pass
# tensor([[  2.98779, 207.45413,  56.88139, 280.08646,   0.78838,   0.00000]], device='cuda:0')


        # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
   
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        print(s)

       

            # Stream results
                im0 = annotator.result()



                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), 640, 640)
                cv2.imshow(str(p), im0)
 
                cv2.waitKey(1)  # 1 millisecond

      

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
