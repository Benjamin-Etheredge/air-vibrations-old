from string import punctuation
import gradio as gr
from transformers import pipeline
from transformers import AutomaticSpeechRecognitionPipeline
from deepmultilingualpunctuation import PunctuationModel

puntuation_model = PunctuationModel()
# capitalization_model = ("KES/caribe-capitalise")
# text = "My name is Clara and I live in Berkeley California Ist das eine Frage Frau Müller"
# print(result)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

capitalise_tokenizer = AutoTokenizer.from_pretrained("KES/caribe-capitalise")
capitalise_model = AutoModelForSeq2SeqLM.from_pretrained("KES/caribe-capitalise")


pipe = pipeline(
    model="facebook/wav2vec2-large-960h", 
    chunk_length_s=180,
    stride_length_s=30,
    device=0)

def translate(audio_file):
    x = pipe(audio_file)
    text = x['text']
    return text

def punctuation(text):
    punctuation = puntuation_model.restore_punctuation(text)
    return punctuation

def capitalise(text):
    text = text.lower()
    inputs = capitalise_tokenizer("text:"+text, truncation=True, return_tensors='pt')
    # print(capitalization)
    output = capitalise_model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
    capitalised_text = capitalise_tokenizer.batch_decode(output, skip_special_tokens=True)

    result = ("".join(capitalised_text))

    return result

def all(file):
    trans_text = translate(file).lower()
    punct_text = punctuation(trans_text)
    cap_text = capitalise(punct_text)
    return trans_text, punct_text, cap_text

input = gr.Audio(type="filepath")
live_in = gr.Audio(type="filepath", source="microphone")
# options = gr.CheckboxGroup(
#     options=["text", "punctuation", "capitalisation"],
# )
raw_output = gr.Text(label="Raw Output")
puncuation_output = gr.Text(label="Punctuation Output")
capitalization_output = gr.Text(label="Capitalization Output")

translater = gr.Interface(
    fn=translate, 
    inputs=input, 
    outputs=[raw_output])

punctuation = gr.Interface(
    fn=punctuation,
    inputs=raw_output,
    outputs=[puncuation_output])

capitalization = gr.Interface(
    fn=capitalise,
    inputs=puncuation_output,
    outputs=[capitalization_output])



# gr.Series(translater, punctuation, capitalization).launch(share=True)
live_demo = gr.Interface(
    fn=all,
    inputs=live_in,
    outputs=[raw_output, puncuation_output, capitalization_output])
demo = gr.Interface(
    fn=all,
    inputs=input,
    outputs=[raw_output, puncuation_output, capitalization_output])

# demo.launch(share=True)
gr.TabbedInterface([demo, live_demo], tab_names=["Upload File", "Record Self"]).launch(share=True)
