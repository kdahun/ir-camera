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

    # Acquire the condition variable and notify the main thread
    # that a new frame is ready to render. This is required since
    # all rendering done by OpenCV needs to happen on the main thread.
    # 블록을 사용하여 프레임을 처리하기 전에 조건 변수를 획득하고, 메인 스레드에 새 프레임이 준비되었음을 알린다.
    with renderer.frame_condition:
        # camera_frame.color_argb8888를 사용하여 새로운 프레임을 가져와서 renderer.frame에 할당
        # 나중에 OpenCv를 사용하여 랜더링된다.
        renderer.frame = camera_frame.color_argb8888

        # 이것을 호출하여 메인 스레드에 새 프레임이 준비되었음을 알린다.
        # 이것은 랜더링 작업이 메인 스레드에서 수행되어야 함을 의미한다.
        renderer.frame_condition.notify()

    

# 이 코드는 Seek Thermal 카메라를 젣어하고 열화상 이미지를 랜더링 하는데 사용되는 비동기 콜백함수인 on_event 함수를 정의하고 있다.
# 파라미터
# 1. camera : 이벤트가 발생한 Seek Thermal 카메라의 참조
# 2. event_type :  SeekCameraManagerEvent 열거형을 통해 어떤 종류의 이벤트가 발생있는지 나타낸다.
# 3. event_status : 이벤트가 오류일 경우 해당 오류에 대한 정보가 제공된다.
# 4. renderer : 사용자 정의 데이터로, Renderer객체의 참조로 전달
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
        camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)

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


def main():
    # window_name 변수에 화면에 표시할 창의 이름을 할당한다.
    window_name = "Seek Thermal - Python OpenCV Sample"
    # cv2.nameWindow 함수를 사용하여 OpenCv창을 생성하고 창의 이름을 설정한다.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create a context structure responsible for managing all connected USB cameras.
    # Cameras with other IO types can be managed by using a bitwise or of the
    # SeekCameraIOType enum cases.
    
    # with SeekCameraManger(SeekCameraIOType.USB) as manager 코드를 사용하여 모든 연결된 USB 카메라를 관리하는
    # SeekCameraManager 인스턴스를 생성한다. 인스턴스 : 일반적으로 실행 중인 임의의 프로세스, 클래스의 현재 생성된 오브젝트를 가리킨다.
    # SeekCameraIOType.USB를 인수로 사용하여 USB카메라를 관리하도록 지정한다
    with SeekCameraManager(SeekCameraIOType.USB) as manager:
        # Start listening for events.

        renderer = Renderer()
        # 이벤트 콜백 함수 등록
        # on_event 함수는 카메라 이벤트(연결, 연결해제, 오류 등)가 발생할 때 호출된다
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
                    # img에는 현재 색만 저장되어 있다.
                    img = renderer.frame.data
                    
                    # Resize the rendering window.
                    # renderer.first_frame 플래그를 사용하여 최초 프레임인 경우 창의 크기를 이미지 크기에 맞게 조정한다.
                    if renderer.first_frame:
                        (height, width, _) = img.shape
                        cv2.resizeWindow(window_name, width * 2, height * 2)
                        renderer.first_frame = False

                    # Render the image to the window.
                    cv2.imshow(window_name, img)

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

# 렌더링은 컴퓨터 그래픽스와 관련된 개념으로, 텀퓨터 프로그램이 데이터를 시각적인 형태로 변환하고 
# 디스플레이에 표시하는 과정을 의미한다. 여기에서 렌더링은 열화상 이미지를 화면에 표시하는 과정을 가리킨다.

# 이미지 데이터 변환 : 열화상 카메라로부터 캡처한 데이터는 숫자로 이루어진 값들의 배열로 표현된다.
# 이 데이터를 컴퓨터 화면에 표시 가능한 이미지 형태로 변환하는 과정이 필요하다
# 이 변환은 이미지 데이터를 시각적 요소로 바꾸는 것을 의미한다.

# 화면 크기 조정 : 이미지 데이터의 크기는 카메라의 해상도와 일치하며, 일반적으로 화면 크기와 다르다.
# 따라서 이미지를 화면에 맞게 크기를 조정하는 과정이 필요하다.

# 화면에 표시 : 변환된 이미지는 화면에 표시된다. 이것이 실제로 사용자에게 보여지는 단계이다.
# 여기서 이미지는 디스플레이에 렌더링 된다.

# 렌더링은 주로 컴퓨터 그래픽스, 게임 개발, 영화 제작, 가상현실, 증강현실 등등에 사용
# 열화상 이미지를 화면에 표시하는 것도 렌더링의 한 형태로 볼 수 있다.
# 이렇게 렌더링된 이미지는 사용자에게 시각적 정보를 제공하거나 분석하는 데 사용된다.