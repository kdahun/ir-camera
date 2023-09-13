#!/usr/bin/env python3
# Copyright 2021 Seek Thermal Inc.
#
# Original author: Michael S. Mead <mmead@thermal.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 렌더러에 대한 액세스를 동기화하는 데 사용되는 Condition 객체를 제공한다.
from threading import Condition

# 이 라이브러리는 열화상 이미지를 표시하는 데 사용되는 openCV 함수를 제공한다.
import cv2
import numpy as np

# 이 라이브러리는 Seek Thermal 카메라를 제어하는 데 사용되는 함수를 제공한다,
from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
    SeekFrame,
)


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

# 열화상 이미지에 색상 매핑을 적용하는 함수
# 1. 최소 및 최대 온도 범위 설정 : min_temperature과 max_temperature 변수를 사용하여 열화상 이미지에서 표시할 온도 범위를 설정
# 2. 정규화 : np.interp 함수를 사용하여 열화상 이미지의 픽셀 값 범위를 변경.
# 정규화를 통해 최소 및 최대 온도 범위 내에서의 온도 값을 0에서 255 범위로 변환한다.
# 3. 정규화된 이지리를 8비트로 변환 : normalized_frame을 np.uint8 데이터 타입으로 변환하여 8비트 이미지로 만든다.
# 이렇게 하면 각 픽셀 값을 0에서 255 사이의 정수로 표현
# 4. 컬러 맵 선택 및 적용 : 선택한 컬러 맵을 cv2.applyColorMap 함수를 사용하여 정규화된 이미지에 적용한다.
# 이 코드에서는 cv2.COLORMAP_JET을 사용하여 열화상 데이터를 컬러로 표현.
# 3. 컬러 이미지 변환 : 최종적으로 컬러로 된 열화상 이미지를 반환한다.
def apply_colormap(frame):
    # 열화상 이미지의 최소 및 최대 온도 범위 설정 (예: 0°C에서 100°C)
    min_temperature = -10.0
    max_temperature = 40.0

    # 열화상 이미지를 8비트로 변환 (0-255 범위)
    normalized_frame = np.interp(frame, (min_temperature, max_temperature), (0, 255))
    normalized_frame = normalized_frame.astype(np.uint8)
    # normalized => 새로 150 가로 200
    print(normalized_frame.shape)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_frame = clahe.apply(normalized_frame)

    # cv2.COLORMAP_JET 또는 다른 컬러 맵을 선택하여 적용
    colormap = cv2.COLORMAP_WINTER
    colored_frame = cv2.applyColorMap(enhanced_frame, colormap)

    return colored_frame

def main():
    window_name = "Seek Thermal - Python OpenCV Sample"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with SeekCameraManager(SeekCameraIOType.USB) as manager:

        renderer = Renderer()
        manager.register_event_callback(on_event, renderer)

        # 무한 루프 => 이 루프는 프레임을 지속적으로 수집하고 렌더링 한다.
        while True:
            # Wait a maximum of 150ms for each frame to be received.
            # A condition variable is used to synchronize the access to the renderer;
            # it will be notified by the user defined frame available callback thread.
            
            # with renderer.frame_condition : 블록 내에서 새로운 프레임을 기다리며
            with renderer.frame_condition:
                # 최대 150ms동안 새 프레임을 기다린다.
                if renderer.frame_condition.wait(150.0 / 1000.0):
                    # 콜백함수에서 가져온 이미지 데이터를 img 변수에 저장
                    img = renderer.frame.data

                    # 열화상 이미지에 색상 매핑 적용
                    colored_img = apply_colormap(img)

                    # colored_img의 형태를 확인하여 적절한 방식으로 처리
                    if len(colored_img.shape) == 3:  # 이미지가 채널 정보를 가지고 있다면
                        (height, width, _) = colored_img.shape
                    else:
                        (height, width) = colored_img.shape

                    if renderer.first_frame:
                        cv2.resizeWindow(window_name, width * 2, height * 2)
                        renderer.first_frame = False

                    # Render the image to the window.
                    cv2.imshow(window_name, colored_img)

            # Process key events.
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            # Check if the window has been closed manually.
            if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                break

    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    main()
