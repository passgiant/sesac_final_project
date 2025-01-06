import gradio as gr
from PIL import Image
import numpy as np

import os
import sys

# 상위 폴더 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

sys.path.append('/home/jupyter/ComfyUI')

from pixar_4_gpu_d import process_image_from_array
from flat2D_ani_4_gpu_d import process_image
from ani_re_4_gpu_d import process_image_from_array as process_image_from_array2
from ani_beauty_real_4_gpu_d import process_image_from_array as process_image_from_array3

from accessory_filter import main as accessory_main
from background_filter import main as background_main

def process_input(input_text, option, webcam_image):
    if webcam_image is not None:
        if isinstance(webcam_image, dict) and "composite" in webcam_image:
            return webcam_image["composite"]  # 최종 편집된 이미지 반환
        else:
            return webcam_image  # 업로드된 이미지를 그대로 반환
    else:
        return None  # 입력이 없으면 출력 없음

def process_accessory(input_image, selected_image):
    """
    입력 이미지와 선택된 액세서리 이미지를 합성하는 함수.

    Args:
        input_image: 입력 이미지 (numpy 배열 또는 PIL 이미지).
        selected_image: 액세서리 이미지 경로.

    Returns:
        최종 합성된 이미지 또는 오류 메시지.
    """
    # 입력 이미지와 액세서리 이미지 확인
    if input_image is None or selected_image is None:
        return "입력 이미지와 선택된 액세서리 이미지를 모두 제공해야 합니다."

    try:
        # 입력 이미지가 numpy 형식이면 PIL 형식으로 변환
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        # accessory_main 함수를 사용하여 액세서리를 처리
        result_image = accessory_main(input_image, selected_image)

        # Gradio 출력 형식에 맞게 NumPy 배열로 변환하여 반환
        result_array = np.array(result_image)
        return result_array

    except Exception as e:
        print(f"에러 발생: {str(e)}")  # 에러 메시지 로깅
        return f"액세서리 처리 중 오류가 발생했습니다: {str(e)}"

# 모델에 따른 함수 실행
def process_model(input_text, selected_option, webcam_image):
    # print(f"webcam_image 타입: {type(webcam_image)}")
    if selected_option == "픽사 스타일 변환 모델":
        return process_image_from_array(webcam_image)
    elif selected_option == "스케치 애니 스타일 변환 모델":
        return process_image(webcam_image)
    elif selected_option == "애니메이션 스타일 변환 모델":
        return process_image_from_array2(webcam_image)
    elif selected_option == "beauty real ani 변환 모델":
        return process_image_from_array3(webcam_image)

# 페이지 전환 함수
def toggle_pages(page_to_show):
    if page_to_show == "page1":
        return gr.update(visible=True), gr.update(visible=False)
    elif page_to_show == "page2":
        return gr.update(visible=False), gr.update(visible=True)

def toggle_editor_with_image(current_page, current_image):
    if current_page == "webcam":  # 현재 웹캠 모드에서 그림판으로 전환
        return (
            gr.update(visible=False, label="웹캠 입력"),  # 웹캠 숨기기
            gr.update(visible=True, label="그림판 입력", value=current_image),  # 그림판 보이기 및 현재 이미지를 설정
            "editor"  # 상태를 그림판으로 설정
        )
    elif current_page == "editor":  # 현재 그림판 모드에서 웹캠으로 전환
        return (
            gr.update(visible=True, label="웹캠 입력", value=current_image),  # 웹캠 보이기 및 현재 이미지를 설정
            gr.update(visible=False, label="그림판 입력"),  # 그림판 숨기기
            "webcam"  # 상태를 웹캠으로 설정
        )

# image_files = ["/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg", "/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg", "/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg", "/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg"]

# 이미지 파일 경로와 표시 이름 매핑
image_map = {
    "2025_headband.jpg": "/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg",
    "christmas_headband_3.jpg": "/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg",
    "newyear_headband.jpg": "/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg",
    "santa_hat.jpg": "/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg"
}

from PIL import Image
import numpy as np

def resize_image(image):
    """
    이미지를 최대 크기 제한으로 리사이즈합니다.
    Args:
        image: 입력 이미지 (dict 또는 numpy.ndarray).
    Returns:
        numpy.ndarray: 리사이즈된 이미지.
    """
    # dict 형태일 경우 composite 키를 통해 이미지 추출
    if isinstance(image, dict):
        # print("Dict 데이터 구조:", image.keys())  # 키 확인
        if "composite" in image:
            image = image["composite"]  # composite 키에서 이미지 추출
        else:
            raise ValueError("올바른 composite 키가 이미지 데이터에 없습니다.")

    # NumPy 배열을 PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        raise ValueError("이미지 데이터가 NumPy 배열이 아닙니다.")

    # 이미지 크기 조정
    max_size = 600  # 최대 크기
    pil_image.thumbnail((max_size, max_size))  # 크기를 제한
    return np.array(pil_image)  # PIL 이미지 -> NumPy 배열 변환

# def dummy_debug_function(img, prompt):
#     print("Dummy function called")
#     print(f"Image type: {type(img)}, Image shape: {getattr(img, 'shape', 'Unknown shape')}")
#     print(f"Prompt: {prompt}")
#     return img  # 입력 이미지를 그대로 반환

# def dummy_debug_output(output_image):
#     print(f"Output Image type: {type(output_image)}, Shape: {getattr(output_image, 'shape', 'Unknown shape')}")
#     return output_image

# Gradio UI 구성
with gr.Blocks() as demo:
    # 헤더 (중앙 정렬 적용)
    gr.Markdown(
        """
        <div style="text-align: center; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
            <h1 style="color: #4682b4;">✨ 드림픽처스 ❄️</h1>
            <h3 style="color: #4682b4;">인생네컷과 스노우를 합한 서비스</h3>
        </div>
        """, 
        elem_id="header"
    )

    # 버튼 레이아웃
    with gr.Row():
        btn_page1 = gr.Button("이펙트 추가")
        btn_page2 = gr.Button("스타일 변환")
    
    # 현재 상태를 저장하기 위한 State
    current_page = gr.State(value="webcam")  # 초기값은 "webcam"

    # 메인 컨텐츠
    with gr.Column(scale=4):
        with gr.Group(visible=True) as page1:
            with gr.Tabs():
                with gr.TabItem("장신구 추가 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/add_acc_ori.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/add_acc_tr.png", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("배경 변환 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/ch_bg_ori.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/ch_bg_tr.png", label="변환된 이미지", height=400, width=600)
                        
            # with gr.Row():
            #     gr.Image(value="/home/jupyter/ComfyUI/my/gradio/aquarium-3461_256.gif", label='사용 설명 움짤', height=450, width=650)

            # with gr.Row():
            #     option1 = gr.Dropdown(
            #         ["장신구 추가 모델", "배경 변환 모델"], 
            #         label="모델 선택"
            #     )
            
            with gr.Row():
                btn_toggle = gr.Button("웹캠/이미지 전환")

                
            # 웹캠과 그림판 (전환 가능)
            with gr.Row():
                webcam_component = gr.Image(label="웹캠 입력", type="numpy", height=891, width=480, visible=True) # height 640 -> 849 -> 891
                editor_component = gr.ImageEditor(label="그림판", type="numpy", height=891, width=480, interactive=True, visible=False) # height 640 -> 849 > 891
                # 이미지 출력 칸
                # output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)
                
                # image_views = [gr.Image(value=img, label=f"Image {i+1}", interactive=False) for i, img in enumerate(image_files)]
                
            # 버튼 정의
            process_button = gr.Button("이미지 리사이즈(그림판 전환 후 진행)")
                
            with gr.Row():
                image1 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg", label="2025_headband", interactive=False)
                image2 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg", label="christmas_headband", interactive=False)
                image3 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg", label="newyear_headband", interactive=False)
                image4 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg", label="santa_hat", interactive=False)
            with gr.Row():
                radio = gr.Radio(choices=list(image_map.keys()), label="이미지 선택", type="value")
                    
            with gr.Row():
                mask_component = gr.Image(label="마스킹 입력", type="numpy", height=640, width=480, visible=True)

            # 버튼 클릭 시 이미지 크기 조정 후 그림판에 표시
            process_button.click(
                lambda img: gr.update(value=resize_image(img)),
                inputs=editor_component,
                outputs=editor_component
            )
            
            btn_toggle.click(
                toggle_editor_with_image,
                inputs=[current_page, webcam_component],  # 추가: webcam_component가 현재 이미지를 입력으로 사용
                outputs=[webcam_component, editor_component, current_page]
            )
            
            input_text = gr.Textbox(label="프롬프트 입력(배경 변환에 사용)", placeholder="입력하세요", visible=True)
            
            with gr.Row():
                btn_toggle = gr.Button("웹캠/이미지 전환")
            
            with gr.Row():
                submit_btn1 = gr.Button("장신구 추가(이미지)")
                submit_btn2 = gr.Button("배경 변환(웹캠)")
            
            # 이미지 출력 칸
            output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)

            # 버튼 클릭 이벤트 (웹캠 또는 그림판의 이미지를 얻어와 처리)
            submit_btn1.click(
                fn=accessory_main,
                inputs=[radio, editor_component, mask_component],  # radio를 입력으로 사용
                outputs=output_image
            )
            
            submit_btn2.click(
                fn=background_main,  # main 함수 호출
                inputs=[webcam_component, input_text],  # Gradio 컴포넌트에서 입력 받음
                outputs=output_image  # 처리된 이미지를 출력
            )
            
            # submit_btn2.click(
            #     fn=dummy_debug_output,
            #     inputs=[output_image],
            #     outputs=output_image
            # )

        with gr.Group(visible=False) as page2:
            with gr.Tabs():
                with gr.TabItem("픽사 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/jenny_ori.webp", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/jenny_tr.webp", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("스케치 애니 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer5.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer_flat2D_ani.webp", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("애니메이션 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/jisoo_ori.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/jisoo_tr.webp", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("beauty real ani 스타일"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer4.JPG", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer_ani_beauty_real.webp", label="변환된 이미지", height=400, width=600)

            with gr.Row():
                option2 = gr.Dropdown(
                    ["픽사 스타일 변환 모델", "스케치 애니 스타일 변환 모델", "애니메이션 스타일 변환 모델", "beauty real ani 변환 모델"], 
                    label="모델 선택"
                )

            with gr.Row():
                webcam_image = gr.Image(label="웹캠 입력", type="numpy", height=640, width=480)

            submit_btn2 = gr.Button("이미지 변환")

            with gr.Row():
                output_image1 = gr.Image(label="출력 이미지 1", type="numpy", height=450, width=650, interactive=False)
                output_image2 = gr.Image(label="출력 이미지 2", type="numpy", height=450, width=650, interactive=False)
            # with gr.Row():
                output_image3 = gr.Image(label="출력 이미지 3", type="numpy", height=450, width=650, interactive=False)
                output_image4 = gr.Image(label="출력 이미지 4", type="numpy", height=450, width=650, interactive=False)

            submit_btn2.click(
                process_model, 
                inputs=[input_text, option2, webcam_image],
                outputs=[output_image1, output_image2, output_image3, output_image4]
            )


        # 버튼 클릭 이벤트
        btn_page1.click(lambda: toggle_pages("page1"), inputs=None, outputs=[page1, page2])
        btn_page2.click(lambda: toggle_pages("page2"), inputs=None, outputs=[page1, page2])

    # 하단 푸터
    gr.Markdown(
        """
        <div style="text-align: center; background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
            <p style="color: #4682b4;">© 2024 인생스노우. All rights reserved.</p>
        </div>
        """
    )

# 실행
demo.launch(share=True)
