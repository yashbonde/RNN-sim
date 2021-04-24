# simple streamlit app for T2R
# @yashbonde - 24/04/2021

import streamlit as st
from t2rmodel import *

def generate(text, num_steps, model, tokenizer):
  input_ids = tokenizer(text, return_tensors = "pt", return_attention_mask=False)["input_ids"]
  generated_seq = model.generate(
    input_ids = input_ids,
    max_length = input_ids.shape[1] + num_steps,
    num_return_sequences = 1
  )
  return tokenizer.decode(generated_seq[0].tolist())


# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation = True)
def get_model():
  model, tokenizer = T2R.from_pretrained("gpt2", feature_size = 32, return_tokenizer = True)
  model.infer_init()
  return model, tokenizer


def main(model, tokenizer):
  st.write('''# Auto Code Generation Demo

  We are looking forward to a future where computer not only solves
  complicated problems but also writes to code to solve those by itself.
  The first step towards such a future is to build neural networks
  that run on CPUs **really** fast.
  ''')

  num_tokens = st.slider("Length of Text to generate", 10, 69, value = 10)
  placeholder ='''def draw_subplots(arrays):
      # function to write subplots'''
  input_text = st.text_area("Input Text", value = placeholder, key = "input_text")

  if st.button("generate") and input_text != placeholder:
    seq = generate(input_text, num_tokens, model, tokenizer)
    seq = f"```\n{seq}\n```"
    st.write(seq, key = "input_text")


if __name__ == "__main__":
  model, tokenizer = get_model()
  main(model, tokenizer)
