import gradio as gr
from transformers import pipeline
from transformers import AutomaticSpeechRecognitionPipeline


pipe = pipeline(
    model="facebook/wav2vec2-large-960h", 
    chunk_length_s=120,
    stride_length_s=15,
    device=0)

def pipeline(audio_file):
    x = pipe(audio_file)
    return x['text']

input = gr.Audio(type="filepath")
output = gr.Text()
demo = gr.Interface(fn=pipeline, inputs=input, outputs=output, )

demo.launch(share=True)