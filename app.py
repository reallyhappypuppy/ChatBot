from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# 대화 히스토리 저장용
chat_history_ids = None
conversation = []  # 사용자와 챗봇의 대화를 저장

@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history_ids, conversation

    if request.method == "POST":
        user_input = request.form["message"]
        conversation.append(("사용자", user_input))

        # 입력 인코딩
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # 히스토리에 이어 붙이기
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # 응답 생성
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        # 응답 디코딩
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        conversation.append(("챗봇", response))

    return render_template("chat.html", conversation=conversation)

if __name__ == "__main__":
    app.run(debug=True)
