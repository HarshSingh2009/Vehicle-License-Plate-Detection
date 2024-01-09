from predict_pipeline import DetectionPipeline
import streamlit as st

st.title('Automatic Vechile LICENSE Plate detection')
st.write('Detects the License plate of a car and predicts the digits present in it! \nPowered by YOLOv8 Medium model')

st.write('')

detect_pipeline = DetectionPipeline()

st.info('License Plate Detector MODEL loaded successfully!')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    with st.container():
        col1, col2 = st.columns([3, 3])
        col1.header('Input Image')
        col1.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        col1.text('')
        col1.text('')

        if st.button('Detect!'):
            preprocessed_img_array = detect_pipeline.preprocess_image(uploaded_file=uploaded_file)
            detections = detect_pipeline.detectLicensePlates(input_array=preprocessed_img_array)
            detections_img = detect_pipeline.detections2Image(preprocess_image=preprocessed_img_array, detections=detections)

            col2.header('Detections')
            col2.image(detections_img, caption='Predictions by model', use_column_width=True)


      