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
def process_model(input_text, selected_option, webcam_image):
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
    
# 탭 전환에 따른 컴포넌트 업데이트 함수
def toggle_tab(tab_name, webcam_image):
    if tab_name == "웹캠 입력":
        # 웹캠 활성화, 그림판 비활성화
        return (
            gr.update(visible=True, value=webcam_image),  # webcam_component 업데이트
            gr.update(visible=False)                      # editor_component 숨기기
        )
    elif tab_name == "그림판 입력":
        # 그림판 활성화, 웹캠 비활성화
        return (
            gr.update(visible=False),                     # webcam_component 숨기기
            gr.update(visible=True, value=webcam_image)   # editor_component 업데이트
        )
    
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
    webcam_visible = gr.State(value=True)
    
    # 버튼 클릭 이벤트 함수 (웹캠 -> 그림판 전환 및 이미지 이동)
    def toggle_editor_with_image(webcam_visible, webcam_image):
        if webcam_visible:  # 현재 웹캠 모드에서 그림판으로 전환
            return (
                gr.update(visible=False, label="웹캠 입력"),  # 웹캠 숨기기
                gr.update(visible=True, label="그림판 입력", value=webcam_image),  # 그림판 보이기 및 이미지 설정
                False  # 상태 업데이트
            )
        else:  # 현재 그림판 모드에서 웹캠으로 전환
            return (
                gr.update(visible=True, label="웹캠 입력"),  # 웹캠 보이기
                gr.update(visible=False, label="그림판 입력"),  # 그림판 숨기기
                True  # 상태 업데이트
            )

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
            
            with gr.Tabs() as tabs:
                # 웹캠 입력 탭
                with gr.TabItem("웹캠 입력"):
                    webcam_component = gr.Image(
                        label="웹캠 입력", type="numpy", height=640, width=480, visible=True
                    )

                # 그림판 입력 탭
                with gr.TabItem("그림판 입력"):
                    editor_component = gr.ImageEditor(
                        label="그림판 입력", type="numpy", height=640, width=480, interactive=True, visible=False
                    )

            # 탭 전환 상태 연결
            tabs.change(
                toggle_tab, 
                inputs=[tabs, webcam_component],  # 현재 탭 이름과 웹캠 이미지를 입력으로 사용
                outputs=[webcam_component, editor_component]  # 업데이트할 컴포넌트들
            )
                
            with gr.Row():
                btn_toggle1 = gr.Button("웹캠 전환")
                btn_toggle2 = gr.Button("이미지 그림판 전환")
                
            # 웹캠과 그림판 (전환 가능)
            with gr.Row():
                webcam_component = gr.Image(label="웹캠 입력", type="numpy", height=640, width=480, visible=True)
                editor_component = gr.ImageEditor(label="그림판", type="numpy", height=640, width=480, interactive=True, visible=False)

                # 오른쪽 이미지 출력 칸
                output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)
            
            # 버튼 클릭 이벤트
            btn_toggle1.click(
                toggle_editor_with_image,
                inputs=[webcam_visible, webcam_component],
                outputs=[webcam_component, editor_component, webcam_visible]
            )
            # 버튼 클릭 이벤트 (웹캠 -> 그림판)
            btn_toggle2.click(
                toggle_editor_with_image,
                inputs=[webcam_visible, webcam_component],  # 웹캠 상태와 이미지를 입력
                outputs=[webcam_component, editor_component, webcam_visible]  # 상태와 UI 업데이트
            )

            # with gr.Row():
            #     webcam_image = gr.ImageEditor(type="numpy", height=640, width=480)
            #     output_image = gr.Image(label="출력 이미지", type="numpy", height=640, width=480, interactive=False)

            input_text = gr.Textbox(label="프롬프트 입력(장신구 추가와 배경 변환에 사용)", placeholder="입력하세요", visible=True)

            submit_btn1 = gr.Button("이미지 변환")

            submit_btn1.click(
                process_model, 
                inputs=[input_text, option1, webcam_component],  # 'webcam_component'로 수정
                outputs=[output_image]
            )

        with gr.Group(visible=False) as page2:
            with gr.Tabs():
                with gr.TabItem("픽사 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer1.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer_pixar.webp", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("스케치 애니 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer5.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer_flat2D_ani.webp", label="변환된 이미지", height=400, width=600)
                with gr.TabItem("애니메이션 스타일 예시"):
                    with gr.Row():
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer3.jpg", label="원본", height=400, width=600)
                        gr.Image(value="/home/jupyter/ComfyUI/my/gradio/qwer_ani.webp", label="변환된 이미지", height=400, width=600)
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
            with gr.Row():
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
