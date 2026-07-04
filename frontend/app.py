import gradio as gr

from api import ask_question


def respond(message, history):
    if history is None:
        history = []

    # Add user message
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    # Get answer from FastAPI
    answer = ask_question(message)

    # Add assistant message
    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return "", history


with gr.Blocks(title="AI IT Helpdesk") as demo:

    gr.Markdown("# 🤖 AI IT Helpdesk")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
        layout="bubble",
        buttons=["copy"],
    )

    textbox = gr.Textbox(
        placeholder="Ask your question...",
        lines=1,
    )

    clear = gr.Button("🗑 Clear Chat")

    textbox.submit(
        fn=respond,
        inputs=[textbox, chatbot],
        outputs=[textbox, chatbot],
    )

    clear.click(
        lambda: (None, []),
        outputs=[textbox, chatbot],
    )

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
)