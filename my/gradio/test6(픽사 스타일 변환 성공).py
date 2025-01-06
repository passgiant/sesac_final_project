import gradio as gr
from PIL import Image
import numpy as np

import os
import sys

# 상위 폴더 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 상위 폴더의 pixar_4_gpu.py에서 process_image 함수 가져오기
from pixar_4_gpu_d import process_image_from_array

def process_input(input_text, option, webcam_image):
    if webcam_image is not None:
        if isinstance(webcam_image, dict) and "composite" in webcam_image:
            return webcam_image["composite"]  # 최종 편집된 이미지 반환
        else:
            return webcam_image  # 업로드된 이미지를 그대로 반환
    else:
        return None  # 입력이 없으면 출력 없음

# 페이지 전환 함수
def toggle_pages(page_to_show):
    if page_to_show == "page1":
        return gr.update(visible=True), gr.update(visible=False)
    elif page_to_show == "page2":
        return gr.update(visible=False), gr.update(visible=True)
    
# Gradio UI 구성
with gr.Blocks() as demo:
    with gr.Row():
        # 왼쪽 사이드바
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align: center; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
                    <h1 style="color: #4682b4;">✨ 드림픽쳐스 ❄️</h1>
                </div>
                """,
                elem_id="header"
            )
            btn_page1 = gr.Button("이펙트 추가")
            btn_page2 = gr.Button("필터 효과")

        # 메인 컨텐츠
        with gr.Column(scale=4):
            with gr.Group(visible=True) as page1: 
                with gr.Row():
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/aquarium-3461_256.gif", label='사용 설명 움짤', height=450, width=650)
                
                with gr.Row():
                    option1 = gr.Dropdown(
                        ["장신구 추가 모델", "배경 변환 모델"], 
                        label="모델 선택"
                    )

                with gr.Row():
                    webcam_image = gr.ImageEditor(type="numpy", height=450, width=650)
                    output_image = gr.Image(label="출력 이미지", type="numpy", height=450, width=650, interactive=False)

                input_text = gr.Textbox(label="프롬프트 입력(장신구 추가와 배경 변환에 사용)", placeholder="입력하세요", visible=True)

                submit_btn = gr.Button("이미지 변환")

                # process_image 함수를 직접 호출하여 연동
                submit_btn.click(
                    lambda image: process_image(image)[0],  # 첫 번째 출력 이미지 반환
                    inputs=[webcam_image], 
                    outputs=[output_image]
                )

            with gr.Group(visible=False) as page2:
                with gr.Tabs():
                    with gr.TabItem("픽사 스타일 예시"):
                        with gr.Row():
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer1.jpg", label="원본", height=400, width=600)
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_pixar.webp", label="변환된 이미지", height=400, width=600)
                    with gr.TabItem("스케치 애니 스타일 예시"):
                        with gr.Row():
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer5.jpg", label="원본", height=400, width=600)
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_flat2D_ani.webp", label="변환된 이미지", height=400, width=600)
                    with gr.TabItem("애니메이션 스타일 예시"):
                        with gr.Row():
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer3.jpg", label="원본", height=400, width=600)
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_ani.webp", label="변환된 이미지", height=400, width=600)
                    with gr.TabItem("beauty real ani 스타일"):
                        with gr.Row():
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer4.JPG", label="원본", height=400, width=600)
                            gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_ani_beauty_real.webp", label="변환된 이미지", height=400, width=600)
                
                with gr.Row():
                    option2 = gr.Dropdown(
                        ["픽사 스타일 변환 모델", "스케치 애니 스타일 변환 모델", "애니메이션 스타일 변환 모델", "beauty real ani 변환 모델"], 
                        label="모델 선택"
                    )

                with gr.Row():
                    webcam_image = gr.Image(label="웹캠 입력", type="numpy", height=450, width=650)
                    
                submit_btn = gr.Button("이미지 변환")

                with gr.Row():
                    output_image1 = gr.Image(label="출력 이미지 1", type="numpy", height=450, width=650, interactive=False)
                    output_image2 = gr.Image(label="출력 이미지 2", type="numpy", height=450, width=650, interactive=False)
                with gr.Row():
                    output_image3 = gr.Image(label="출력 이미지 3", type="numpy", height=450, width=650, interactive=False)
                    output_image4 = gr.Image(label="출력 이미지 4", type="numpy", height=450, width=650, interactive=False)

                submit_btn.click(
                    process_image_from_array, # 4개 이미지 출력
                    inputs=[webcam_image],
                    outputs=[output_image1, output_image2, output_image3, output_image4],
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
