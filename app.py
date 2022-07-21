import streamlit as st
import degirum as dg
from PIL import Image

zoo = dg.connect_model_zoo('dgcps://cs.degirum.com', token=st.secrets['DEGIRUM_CLOUD_TOKEN'])

st.title('DeGirum Cloud Demo for ORCA Models')

all_orca_models=zoo.list_models(device='ORCA',)
model_options=[]
for model_name in all_orca_models:
    if 'yamnet' not in model_name and 'comma' not in model_name:
        model_options.append(model_name)
st.header('Choose and Run a Model')
st.text('Select a model and upload an image. Then click on the submit button')
with st.form("model_form"):
    model_name=st.selectbox("Choose a Model from the list", model_options)
    uploaded_file=st.file_uploader('input image')
    submitted = st.form_submit_button("Submit")
    if submitted:
        model=zoo.load_model(model_name)
        model.overlay_font_scale=3
        model.overlay_line_width=6
        if model.output_postprocess_type=='PoseDetection':
            model.overlay_show_labels=False
        st.write("Model loaded successfully")
        image = Image.open(uploaded_file)
        predictions=model(image)
        if model.output_postprocess_type=='Classification' or model.output_postprocess_type=='DetectionYoloPlates':
            st.image(predictions.image,caption='Original Image')
            st.write(predictions.results)
        else:
            st.image(predictions.image_overlay,caption='Image with Bounding Boxes/Keypoints')
            
