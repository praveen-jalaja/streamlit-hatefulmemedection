import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import save_img
import tensorflow_text as text
import shutil
import numpy as np
import os
import json
import sys
import random
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import wordnet
from functools import partial
data_dir = "/streamlit-hatefulmemedection/"
SLANG_PATH = "static\\slang.txt"
import webbrowser # inbuilt module

image_shape = (256,256,3)
trainable = False
max_seq_length = 128
units = 512
embedding_dim = 768
BATCH_SIZE = 50
BUFFER_SIZE = 200
attention_features_shape = 64

#=================================== Title ===============================
st.title("""
HateFul Meme Detector
	""")

#================================= Title Image ===========================
st.text("""""")
img_path_list = ["static\\image_1.jpg",
				"static\\image_2.jpg"]
index = random.choice([0,1])
image = Image.open(img_path_list[index])
st.image(
	        image,
	        use_column_width=True,
	    )

#================================= About =================================
st.write("""
## 1️⃣ About
	""")
st.write("""
Hi all, Welcome to this project. It is a Hateful Meme Detector App!!!
	""")
st.write("""
You have to upload your own test images to test it!!!
	""")
st.write("""
**Or**, if you don't have any memes with you, then also no problem **(**😎**)**, I already selected some test images with text files for you, you have to just go to that section & click the **⬇️ Download** button to download those pictures!  
	""")

#============================ How To Use It ===============================
st.write("""
## 2️⃣ How To Use It
	""")
st.write("""
Well, it's pretty simple!!!
- Let me clear first, the model has power to any memes with text on it only, so you are requested to give image and text written on the image,if image or text is missing you will prompted again to upload or write it 😆 
- First of all, download Meme image and text on it!
- Next, just Browse that files or Drag & drop that file!
- Please make sure that, you are uploading a picture file and txt file!
- Press the **👉🏼 Predict** button to see the magic!!!

🔘 **NOTE :** *If you upload other than an image file/txt file, then it will show an error massage when you will click the* **👉🏼 Predict** *button!!!*
	""")

#========================= What It Will Predict ===========================
st.write("""
## 3️⃣ What It Will Predict
	""")
st.write("""
Well, it can predict wheather the image you have uploaded is the HateFul 😠 or Not-Hateful 🤪?
	""")

#============================== Sample Images For Testing ==================
st.write("""
## 4️⃣  Download Some Images For Testing!!!
	""")
st.write("""
Hey there! here is some meme images and it's texts on google drive!
- Here you can find a total of 10 images **[**5 for each category**]**
- Just click on **⬇️ Download** button & download those images and text files!!!
- The meme and its corresponding text file have same name. Download both for prediction
- You can also try your own images!!!
	""")

#============================= Download Button =============================
st.text("""""")
download = st.button("⬇️ Download")

#============================ Download Clicked =============================
if download:
	link = "https://drive.google.com/drive/folders/12HJwwTpri-JD4PRqWp6TwqR-1TXe1tVG?usp=sharing"
	try:
		webbrowser.open(link)
	except:
		st.write("""
    		⭕ Something Went Wrong!!! Please Try Again Later!!!
    		""")

#============================ Behind The Scene ==========================
st.write("""
## 5️⃣ Behind The Scene
	""")
st.write("""
To see how it works, please click the button below!
	""")
st.text("""""")
github = st.button("👉🏼 Click Here To See How It Works")
if github:
	github_link = "https://github.com/praveen-jalaja/streamlit-hatefulmemedection"
	try:
		webbrowser.open(github_link)
	except:
		st.write("""
    		⭕ Something Went Wrong!!! Please Try Again Later!!!
    		""")

#======================== Time To See The Magic ===========================
st.write("""
## 👁️‍🗨️ Time To See the Model Prediction 🌀
	""")
#==================================== Model ==================================
##================================Preloading function=======================================
@st.cache()
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

@st.cache()
def download_wornet():
  nltk.download('wordnet')

@st.cache()
def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
    download_wornet()
    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word

with open(SLANG_PATH) as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])


def text_preprocessing(final):
  preprocessed_text = []
  for sentance in final:
      sentance = BeautifulSoup(sentance, 'lxml').get_text()
      sentance = replaceSlang(replaceElongated(sentance))
      sentance = decontracted(sentance)
      sentance = re.sub("\S*\d\S*", "", sentance).strip()
      sentance = re.sub('[^A-Za-z]+', ' ', sentance)
      preprocessed_text.append(sentance.strip())
  return preprocessed_text



@st.cache(allow_output_mutation=True)
def load_model():
  image_model = tf.keras.applications.ResNet152V2(include_top=False,
                                                weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output
  image_features_extract_model = tf.keras.Model(new_input, hidden_layer, name = "image_feature_extractor")


  tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

  tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1"

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing_text_layer')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)

  text_features_extract_model= tf.keras.Model([text_input], [outputs['sequence_output'],outputs['pooled_output']], name = 'text_feature_extractor')

  class CNN_Encoder(tf.keras.Model):
      def __init__(self, embedding_dim):
          super(CNN_Encoder, self).__init__()
          self.fc = tf.keras.layers.Dense(units = embedding_dim,kernel_initializer='glorot_uniform',use_bias=False)

      def call(self, x):
          x = self.fc(x)
          x = tf.nn.relu(x)
          return x


  class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.W1 = tf.keras.layers.Dense(units)
      self.W2 = tf.keras.layers.Dense(units)
      self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
      hidden_with_time_axis = tf.expand_dims(hidden, 1)

      attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                          self.W2(hidden_with_time_axis)))

      score = self.V(attention_hidden_layer)
      attention_weights = tf.nn.softmax(score, axis=1)
      context_vector = attention_weights * features
      context_vector = tf.reduce_sum(context_vector, axis=1)

      return context_vector, attention_weights

  class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units,class_size = 2):
      super(RNN_Decoder, self).__init__()
      self.units = units
      self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
      
      self.fc1 = tf.keras.layers.Dense(self.units)

      self.fc2 = tf.keras.layers.Dense(64, activation='relu', name='Dense_3_layer')

      self.dropout  = tf.keras.layers.Dropout(0.3)

      self.fc3  = tf.keras.layers.Dense(class_size, activation='softmax', name='classifier')

      self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
      context_vector, attention_weights = self.attention(features, hidden)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
      output, state = self.gru(x)
      x = self.fc1(output)
      x = tf.reshape(x, (-1, x.shape[2]))
      x = self.fc2(x)
      x = self.dropout(x)
      x = self.fc3(x)

      return x, state, attention_weights

    def reset_state(self, batch_size):
      return tf.zeros((batch_size, self.units))


  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units,class_size = 2)
  optimizer = tf.keras.optimizers.Adam()

  checkpoint_path = "checkpoints\\train"
  ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

  start_epoch = 0
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

  return image_features_extract_model, text_features_extract_model, encoder, decoder


@st.cache(allow_output_mutation=True)
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img


class_values = ["non-hateful","hateful"]
@st.cache(allow_output_mutation=True)
def evaluate(image,text):
    image_features_extract_model, text_features_extract_model, encoder, decoder = load_model()
    text = text_preprocessing([text])[0]
    attention_plot = np.zeros((1, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    img_tensor = tf.expand_dims(load_image(image), 0)
    img_tensor = image_features_extract_model(img_tensor)
    img_tensor = tf.reshape(img_tensor,
             (img_tensor.shape[0], -1, img_tensor.shape[3]))
    text = tf.convert_to_tensor(text)
    text = tf.expand_dims(text, 0)
    text_features_seq,pooled_features = text_features_extract_model(tf.convert_to_tensor(text))
    dec_input = tf.expand_dims(pooled_features, axis = 1)
    features = encoder(img_tensor)
    predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
    
    predicted_id = np.argmax(predictions, axis = 1).tolist()
    return predictions,class_values[predicted_id[0]]

#========================== File Selector ===================================
# st.write("""
# #### Select from the Drop Down test memes
# """)

# option = st.selectbox('Select a Meme',
#                       [x for x in os.listdir(os.path.join(data_dir+"images")) if '.png' in x ])

# img_file_buffer = data_dir+"images/"+option
# txt_file_buffer = data_dir+"images/"+option.split('.')[0]+'.txt'

# st.write("""**Or**""")

#========================== File Uploader ===================================

img_file_buffer = st.file_uploader("Upload an image(.png preffered) here 👇🏻")
txt_file_buffer = st.file_uploader("Upload an text file(.txt) here 👇🏻")

text = ""
if txt_file_buffer:
  for line in txt_file_buffer:
    text = line

try:
	image = Image.open(img_file_buffer)

	st.write("""
		Preview 👀 Of Given Image!
		""")

	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		Now, you are just one step ahead of prediction.
		""")
	st.write("""
		**Just Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This meme! 😄**
		""")
except:
	st.write("""
		### ❗ Any Picture hasn't selected yet!!!
		""")

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")


##=================prediction================================================#
def generate_result(prediction):
	st.write("""
	## 🎯 RESULT
		""")
	if prediction == "hateful":
	    st.write("""
	    	## Model predicts it as an Hateful Meme  😈!!!
	    	""")
	else:
	    st.write("""
	    	## Model predicts it as an Non-Hateful Meme 🤗!!!
	    	""")

#=========================== Predict Button Clicked ==========================
if submit:

  save_img("test_image.png", np.array(image))

  image_path = "test_image.png"
  # Predicting
  st.write("👁️ Predicting...")
  pred, predicted_class = evaluate(image_path,text)

  generate_result(predicted_class)



#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.write("""
### ©️ Created By Praveen Jalaja
	""")
