import streamlit as st
import degirum as dg
from PIL import Image

zoo = dg.connect_model_zoo('dgcps://cs.degirum.com', token=st.secrets['DEGIRUM_CLOUD_TOKEN'])

st.title('DeGirum Cloud Demo')

st.header('Specify Model Options Below')
precision=st.radio("Choose model precision",("Float","Quant","Don't Care"),index=2)
runtime_agent=st.radio("Choose runtime agent",("TFLite","N2X","Don't Care"),index=2)
precision=precision if precision!="Don't Care" else ""
runtime_agent=runtime_agent if runtime_agent!="Don't Care" else "" 
model_options=zoo.list_models(device='ORCA',precision=precision,runtime=runtime_agent)
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
            
