import torch
import clip
import numpy as numpy
from PIL import Image
import os
import pandas as pd
import urllib
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import urllib

context = pd.read_csv('Context2.csv')
flags_map = {}
for i in context.iterrows():
    flags_map[i[1]['PRIMARY_CONTEXT']]={'DISPLAY':i[1]['PRIMARY_LABEL'],
                                    'CONTEXT':i[1]['PRIMARY_CONTEXT'], 
                                    'THRESHOLD':i[1]['THRESHOLD']}
safe_flag = 'NOT, NO'

#CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

#Content Moderation Function
# @st.cache(suppress_st_warning=True)
def detect_moderation_label(image_path, safe_flag, flags_map, threshold):
    image_processed = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    flags = list(flags_map.keys())
    flags_tokenized = clip.tokenize(flags).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_processed)
        text_features = clip_model.encode_text(flags_tokenized)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        stacked_text_features = []
        for i in range(text_features.shape[0]-1):
            stacked_text_features.append(text_features[[i,-1],:].T.unsqueeze(0))
        stacked_text_features = torch.vstack(stacked_text_features)
        softmax_probs = (100*image_features @ stacked_text_features).softmax(dim=-1)
    max_prob,index = softmax_probs[:,:,0].max().numpy().tolist(), softmax_probs[:,:,0].argmax().numpy()
    if max_prob > threshold:
        return flags_map.get(flags[index]).get('DISPLAY'),max_prob
    return flags_map.get(safe_flag).get('DISPLAY'), threshold

#Generate content moderation data using image path
@st.cache(suppress_st_warning=True)
def generate_moderation_data_using_image_path(image_path):
    label = []
    prob = []
    image_list = os.listdir(image_path)
    for i in image_list:
        try:
            moderation_label,probability = detect_moderation_label(image_path+i,safe_flag, flags_map)
            label.append(moderation_label)
            prob.append(probability)
        except:
            label.append('Unable to retrieve')
            prob.append(0)
    df1 = pd.DataFrame()
    df1['Images_ID'] = image_list
    df1['Predicted Moderation Labels'] = label
    df1['Probabilities'] = prob
    return df1

#Generate content moderation data using url
@st.cache(suppress_st_warning=True)
def generate_moderation_data_using_image_url(image_url):
    label1 = []
    prob1 = []
    for i in image_url:
        try:
            img = urllib.request.urlopen(i)
            moderation_label,probability = detect_moderation_label(img,safe_flag, flags_map)
            label1.append(moderation_label)
            prob1.append(probability)
        except:
            label1.append('Unable to retrieve')
            prob1.append(0)
    df2 = pd.DataFrame()
    df2['Images_ID'] = image_url
    df2['Predicted Moderation Labels'] = label1
    df2['Probabilities'] = prob1
    return df2

def get_probability(img,first_context, second_context):
    image = clip_preprocess(Image.open(img)).unsqueeze(0).to(device)
    text = clip.tokenize([first_context, second_context]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        
        logits_per_image, logits_per_text = clip_model(image, text)
    return logits_per_image.softmax(dim=-1).cpu().numpy()

