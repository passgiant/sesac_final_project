import gradio as gr

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

    with gr.Row():
        gr.Image(value="C:/Users/YJKIM_PC/gradio/aquarium-3461_256.gif", label='사용 설명 움짤', height=400, width=600)

    with gr.Group(visible=True) as page1: 
        # gr.Textbox(code_4, label="인생스노우 사용설명서 및 안내서", lines=7)

        # 입력 섹션 (텍스트 입력과 옵션 선택 한 줄로 배치)
        with gr.Row():
            # input_text = gr.Textbox(label="텍스트 입력", placeholder="입력하세요", visible=True)
            option = gr.Dropdown(
                ["장신구 추가 모델", "배경 변환 모델"], 
                label="모델 선택"
            )

        with gr.Row():
            webcam_image = gr.ImageEditor(type="numpy", height=450, width=650)
            # sketchpad_image = gr.Sketchpad(label="그림판", height=450, width=650)
            # 출력 섹션 (결과를 이미지로 출력)
            output_image = gr.Image(label="출력 이미지", type="numpy", interactive=False)

        input_text = gr.Textbox(label="프롬프트 입력(장신구 추가와 배경 변환에 사용)", placeholder="입력하세요", visible=True)

        # 이미지 변환 버튼 (그 다음 줄)
        submit_btn = gr.Button("이미지 변환")

        # 옵션 선택 시 텍스트 입력창 표시/숨기기
        # option.change(toggle_textbox, inputs=option, outputs=input_text)

        # 처리 버튼 클릭 이벤트
        submit_btn.click(process_input, inputs=[input_text, option, webcam_image], outputs=[output_image])

    with gr.Group(visible=False) as page2:

        # gr.Textbox(code_snow, label="인생스노우 사용설명서 및 안내서", lines=7)

        # 탭 섹션
        with gr.Tabs():
            with gr.TabItem("픽사 스타일 예시"):
                with gr.Row():
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer1.jpg", label="원본", height=300, width=500)
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_pixar.webp", label="변환된 이미지", height=300, width=500)
            with gr.TabItem("스케치 애니 스타일 예시"):
                with gr.Row():
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer5.jpg", label="원본", height=300, width=500)
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_flat2D_ani.webp", label="변환된 이미지", height=300, width=500)
            with gr.TabItem("애니메이션 스타일 예시"):
                with gr.Row():
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer3.jpg", label="원본", height=300, width=500)
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_ani.webp", label="변환된 이미지", height=300, width=500)
            with gr.TabItem("beauty real ani 스타일"):
                with gr.Row():
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer4.JPG", label="원본", height=300, width=500)
                    gr.Image(value="C:/Users/YJKIM_PC/gradio/qwer_ani_beauty_real.webp", label="변환된 이미지", height=300, width=500)

        # 입력 섹션 (텍스트 입력과 옵션 선택 한 줄로 배치)
        with gr.Row():
            # input_text = gr.Textbox(label="텍스트 입력", placeholder="입력하세요", visible=True)
            option = gr.Dropdown(
                ["픽사 스타일 변환 모델", "스케치 애니 스타일 변환 모델", "애니메이션 스타일 변환 모델", "beauty real ani 변환 모델"], 
                label="모델 선택"
            )

        # 웹캠 입력과 그림판 (같은 줄에 배치)
        with gr.Row():
            webcam_image = gr.Image(label="웹캠 입력", type="numpy", height=450, width=650)
            # 출력 섹션 (결과를 이미지로 출력)
            output_image = gr.Image(label="출력 이미지", type="numpy", height=450, width=650, interactive=False)

        with gr.Row():
            submit_btn = gr.Button("이미지 변환")

        # 옵션 선택 시 텍스트 입력창 표시/숨기기
        # option.change(toggle_textbox, inputs=option, outputs=input_text)

        # 처리 버튼 클릭 이벤트
        submit_btn.click(process_input, inputs=[input_text, option, webcam_image], outputs=[output_image])

    # 버튼 클릭 이벤트
    btn_page1.click(lambda: toggle_pages("page1"), inputs=None, outputs=[page1, page2])
    btn_page2.click(lambda: toggle_pages("page2"), inputs=None, outputs=[page1, page2])

    # 하단 푸터 (중앙 정렬 적용)
    gr.Markdown(
        """
        <div style="text-align: center; background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
            <p style="color: #4682b4;">© 2024 인생스노우. All rights reserved.</p>
        </div>
        """
    )

# 실행
demo.launch(share=True)