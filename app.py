#Importing libraries
import streamlit as st
from PIL import Image
import urllib
import pandas as pd
from Utils import detect_moderation_label
from Utils import generate_moderation_data_using_image_path, generate_moderation_data_using_image_url, get_probability
import warnings
from Utils import flags_map, safe_flag
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------------------------------------------

#Adding Content moderation Title
st.sidebar.title('MODERATION APPLICATION')

#Adding header
st.header('''IMAGE MODERATION''')

#Selecting url or image
image_source_type = st.sidebar.radio('Please select below:',['About',
                                                             'Single JPG Image',
                                                             'Single URL Image',
                                                             'Single URL Image with two context'])
##############################################################################################################
if image_source_type=='About':

    st.write('This application can detect explicit and graphical content in the image.')
    
    st.write('\n')

    st.subheader('Moderation Flags:')
    st.write('`1. EXPLICIT NUDITY & SUGGESTIVE`')
    st.write('`2. ALCOHOL`')
    st.write('`3. DRUGS & TOBACCO`')
    st.write('`4. VIOLENCE & VISUALLY DISTURBING`')
    st.write('`5. RUDE GESTURES & HATE SYMBOLS`')
    st.write('`6. GAMBLING`')
    st.write('`7. NEUTRAL`')

##############################################################################################################
elif image_source_type=='Single JPG Image':
    
    #Taking image input through upload
    image_input = st.file_uploader('Moderation on single JPG image:', type=['jpg'])

    threshold = st.slider('Probability threshold:',0.51,0.99,0.1)

    #Reading image
    try:
        img = Image.open(image_input)
        labels, probs = detect_moderation_label(image_input, safe_flag, flags_map, threshold)
    except:
        st.markdown("""###### Please upload image & click below""")

    #Content Moderation resutls
    if st.button('MODERATE IMAGE'):
        st.image(img, width=400)
        st.metric(label='Moderation Label:', value=labels)
        st.metric(label='Moderation Probability:', value=f'{round(probs*100,2)}%')

##############################################################################################################
elif image_source_type=='Single URL Image':
    #Taking image input through url
    image_input = st.text_input('Moderation on single URL image:')

    threshold = st.slider('Probability threshold:',0.51,0.99,0.1)
    #Reading image url
    try:
        img = urllib.request.urlopen(image_input)
        labels, probs = detect_moderation_label(img, safe_flag, flags_map, threshold)
    except:
        st.markdown("""###### Please upload image URL & click below""")

   #Content Moderation resutls
    if st.button('MODERATE IMAGE'):
        st.image(image_input, width=400)  
        st.metric(label='Moderation Label:', value=labels)
        st.metric(label='Moderation Probability:', value=f'{round(probs*100,2)}%')

##############################################################################################################
else:
    #Taking image input through url
    image_input = st.text_input('Moderation on single URL image:')
    first_context = st.text_input('First Context:')
    second_context = st.text_input('Second Context:')
    #Reading image url
    try:
        img = urllib.request.urlopen(image_input)
        probs = get_probability(img, first_context, second_context)
    except:
        st.markdown("""###### Please upload image URL & click below""")

   #Content Moderation resutls
    if st.button('EXTRACT PROBABILITY'):
        st.image(image_input, width=400)  
        st.metric(label=first_context.upper(), value=f'{round(probs[0][0]*100,2)}%')
        st.metric(label=second_context.upper(), value=f'{round(probs[0][1]*100,2)}%')
        
        


       



