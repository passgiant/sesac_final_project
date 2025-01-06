import gradio as gr
from PIL import Image
import numpy as np

import os
import sys

# 상위 폴더 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pixar_4_gpu_d import process_image_from_array
from flat2D_ani_4_gpu_d import process_image
from ani_re_4_gpu_d import process_image_from_array as process_image_from_array2
from ani_beauty_real_4_gpu_d import process_image_from_array as process_image_from_array3

def process_input(input_text, option, webcam_image):
    if webcam_image is not None:
        if isinstance(webcam_image, dict) and "composite" in webcam_image:
            return webcam_image["composite"]  # 최종 편집된 이미지 반환
        else:
            return webcam_image  # 업로드된 이미지를 그대로 반환
    else:
        return None  # 입력이 없으면 출력 없음

# 모델에 따른 함수 실행
def process_model2(input_text, selected_option, webcam_image):
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
    
# # 버튼 클릭 이벤트 함수 (웹캠 -> 그림판 전환 및 이미지 이동)
# def toggle_editor_with_image(webcam_visible, webcam_image):
#     if webcam_visible:  # 현재 웹캠 모드에서 그림판으로 전환
#         return (
#             gr.update(visible=False, label="웹캠 입력"),  # 웹캠 숨기기
#             gr.update(visible=True, label="그림판 입력", value=webcam_image),  # 그림판 보이기 및 이미지 설정
#             False  # 상태 업데이트
#         )
#     else:  # 현재 그림판 모드에서 웹캠으로 전환
#         return (
#             gr.update(visible=True, label="웹캠 입력"),  # 웹캠 보이기
#             gr.update(visible=False, label="그림판 입력"),  # 그림판 숨기기
#             True  # 상태 업데이트
#         )
    
# 버튼 클릭 이벤트 함수 (웹캠 -> 그림판 전환 및 이미지 이동)
def toggle_editor_with_image(current_page):
    if current_page == "webcam":  # 현재 웹캠 모드에서 그림판으로 전환
        return (
            gr.update(visible=False, label="웹캠 입력"),  # 웹캠 숨기기
            gr.update(visible=True, label="그림판 입력"),  # 그림판 보이기
            "editor"  # 상태를 그림판으로 설정
        )
    elif current_page == "editor":  # 현재 그림판 모드에서 웹캠으로 전환
        return (
            gr.update(visible=True, label="웹캠 입력"),  # 웹캠 보이기
            gr.update(visible=False, label="그림판 입력"),  # 그림판 숨기기
            "webcam"  # 상태를 웹캠으로 설정
        )    

# image_files = ["/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg", "/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg", "/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg", "/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg"]

# 이미지 파일 경로와 표시 이름 매핑
image_map = {
    "2025_headband": "/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg",
    "christmas_headband": "/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg",
    "newyear_headband": "/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg",
    "santa_hat": "/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg"
}

def select_image(selected_name):
    # 선택한 이름에 해당하는 실제 이미지 경로 반환
    selected_image_path = image_map[selected_name]
    return f"선택한 이미지: {selected_name}"


# Gradio UI 구성
with gr.Blocks() as demo:
    # 헤더 (중앙 정렬 적용)
    gr.Markdown(
        """
        <div style="text-align: center; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
            <h1 style="color: #4682b4;">✨ 드림픽쳐스 ❄️</h1>
            <h3 style="color: #4682b4;">인생네컷과 스노우를 합한 서비스</h3>
        </div>
        """, 
        elem_id="header"
    )

    # 버튼 레이아웃
    with gr.Row():
        btn_page1 = gr.Button("이펙트 추가")
        btn_page2 = gr.Button("필터 효과")
        
    # with gr.Row():
    #     # 왼쪽 사이드바
    #     with gr.Column(scale=1):
    #         gr.Markdown(
    #             """
    #             <div style="text-align: center; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
    #                 <h1 style="color: #4682b4;">✨ 드림픽쳐스 ❄️</h1>
    #             </div>
    #             """,
    #             elem_id="header"
    #         )
    #         btn_page1 = gr.Button("이펙트 추가")
    #         btn_page2 = gr.Button("필터 효과")
    
    # 웹캠과 그림판 전환 상태를 저장
    # webcam_visible = gr.State(value=True)
    
    # 현재 상태를 저장하기 위한 State
    current_page = gr.State(value="webcam")  # 초기값은 "webcam"

    # 메인 컨텐츠
    with gr.Column(scale=4):
        with gr.Group(visible=True) as page1: 
            with gr.Row():
                gr.Image(value="/home/jupyter/ComfyUI/my/gradio/aquarium-3461_256.gif", label='사용 설명 움짤', height=450, width=650)

            with gr.Row():
                option1 = gr.Dropdown(
                    ["장신구 추가 모델", "배경 변환 모델"], 
                    label="모델 선택"
                )
                
            # with gr.Row():
            #     btn_toggle1 = gr.Button("웹캠 전환")
            #     btn_toggle2 = gr.Button("이미지 그림판 전환")
            
            with gr.Row():
                btn_toggle = gr.Button("웹캠/그림판 전환")

                
            # 웹캠과 그림판 (전환 가능)
            with gr.Row():
                webcam_component = gr.Image(label="웹캠 입력", type="numpy", height=849, width=480, visible=True) # height 640
                editor_component = gr.ImageEditor(label="그림판", type="numpy", height=849, width=480, interactive=True, visible=False) # height 640

                # 이미지 출력 칸
                # output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)
                
                # image_views = [gr.Image(value=img, label=f"Image {i+1}", interactive=False) for i, img in enumerate(image_files)]
                
                # 1. 이미지 2줄로 나누어 배치
                with gr.Column():
                    with gr.Row():  # 첫 번째 줄
                        image1 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/2025_headband.jpg", label="2025_headband", interactive=False)
                        image2 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/christmas_headband_3.jpg", label="christmas_headband", interactive=False)
                    with gr.Row():  # 두 번째 줄
                        image3 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/newyear_headband.jpg", label="newyear_headband", interactive=False)
                        image4 = gr.Image(value="/home/jupyter/ComfyUI/my/gradio/santa_hat.jpg", label="santa_hat", interactive=False)
                
                    # 2. 라디오 버튼으로 사용자 지정 이름 표시
                    radio = gr.Radio(choices=list(image_map.keys()), label="이미지 선택", type="value")
            
            # 선택 결과 표시
            output = gr.Textbox(label="결과")
            
            # 선택 버튼
            select_button = gr.Button("이미지 선택 확인")
            select_button.click(select_image, inputs=radio, outputs=output)
            
            # # 버튼 클릭 이벤트
            # btn_toggle1.click(
            #     toggle_editor_with_image,
            #     inputs=[webcam_visible, webcam_component],
            #     outputs=[webcam_component, editor_component, webcam_visible]
            # )
            # # 버튼 클릭 이벤트 (웹캠 -> 그림판)
            # btn_toggle2.click(
            #     toggle_editor_with_image,
            #     inputs=[webcam_visible, webcam_component],  # 웹캠 상태와 이미지를 입력
            #     outputs=[webcam_component, editor_component, webcam_visible]  # 상태와 UI 업데이트
            # )
            
            # 버튼 클릭 이벤트 (웹캠 <-> 그림판 전환)
            btn_toggle.click(
                toggle_editor_with_image,
                inputs=current_page,
                outputs=[webcam_component, editor_component, current_page]
            )

            # with gr.Row():
            #     webcam_image = gr.ImageEditor(type="numpy", height=640, width=480)
            #     output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)

            input_text = gr.Textbox(label="프롬프트 입력(장신구 추가와 배경 변환에 사용)", placeholder="입력하세요", visible=True)

            submit_btn1 = gr.Button("이미지 변환")
            
            # 이미지 출력 칸
            output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)

            submit_btn1.click(
                process_model2, 
                inputs=[input_text, option1, webcam_component],  # 'webcam_component'로 수정
                outputs=[output_image]
            )

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
                process_model2, 
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