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

#Adding Content moderation Title
st.sidebar.title('MODERATION APPLICATION')

#Adding header
st.header('''CONTENT MODERATION USING CLIP MODEL''')

#Selecting url or image
image_source_type = st.sidebar.radio('Please select below:',['Single JPG Image',
                                                             'Single URL Image', 
                                                             'Multiple JPG Images',
                                                             'Multiple URL Images',
                                                             'Single JPG Image with two context'])
##############################################################################################################
if image_source_type=='Single JPG Image':
    
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
elif image_source_type=='Multiple JPG Images':
    #Taking multiple image through image path
    image_input = st.text_input('Moderation on multiple JPG image:')

    threshold = st.slider('Probability threshold:',0.51,0.99,0.1)

    #Generate moderation data
    try:
        moderation_data = generate_moderation_data_using_image_path(image_input, threshold)
        df = moderation_data.to_csv()
    except:
        st.markdown("""###### Please provide image folder path & click below""")

    #Generate content moderation data and prfloat on screen
    if st.button('GENERATE MODERATION DATA'):
        st.dataframe(moderation_data.head(10))
        #Download the data
        st.download_button('DOWNLOAD',data=df,file_name='Moderation_Data.csv',)
##############################################################################################################
elif image_source_type=='Multiple URL Images':
    #Taking multiple image through image url csv file
    image_input = st.file_uploader('Moderation on multiple URL image:', type=['csv'])
    if image_input is not None:
        image_data = pd.read_csv(image_input)

    threshold = st.slider('Probability threshold:',0.51,0.99,0.5)

    #Generate moderation data
    try:
        moderation_data_url = generate_moderation_data_using_image_url(image_data['Images_ID'], threshold)
        df = moderation_data_url.to_csv()
    except:
        st.markdown("""###### Please provide multiple images url csv file & click below""")

    # #Generate content moderation data and prfloat on screen
    if st.button('GENERATE MODERATION DATA'):
        st.write(f'Data Shape: {image_data.shape}')
        st.dataframe(moderation_data_url.head())
    
        #Download the data
        st.download_button('DOWNLOAD',data=df,file_name='Moderation_Data.csv')
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
        
        


       



