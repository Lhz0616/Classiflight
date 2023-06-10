import streamlit as st
import fasttext as ft
import joblib
import pickle

from utils import vectorise, tokenise_sentence

FASTTEXT_MODEL_PATH = './models/fasttext_text_classification_model.bin'
WORD2VEC_MODEL_PATH = './models/svm_model.sav'


with st.spinner('Wait for models to load...'):
    fasttext_model = ft.load_model(FASTTEXT_MODEL_PATH)
    word2vec_model = joblib.load(WORD2VEC_MODEL_PATH)
    pkl_file = open('./models/classes.pkl', 'rb')
    label_encoder = pickle.load(pkl_file)
    pkl_file.close()
    success = st.success('Successfully load models!')
success.empty()


def set_stage(stage):
    st.session_state.stage = stage


if 'stage' not in st.session_state:
    st.session_state.stage = 0

with st.sidebar:
    st.markdown("""
    ## Classification Of Issues regarding Flight Operations Based On Customer Questions 
    
    We are Classiflight, a group of Year 3 Student's that are trying to finish their assignments. :smirk:
    
    In this project, we are focusing on the transportation field, the airline service. 
    I believe that a lot of people travel around every year and there are a lot of people travelling around the world. 
    
    Thus, there are a lot of problems that will be ask by the customers. There is a need for the classification of the questions
    ask by the customers.
    
    We had trained two different models in our project, which is **fasttext** and **Word2Vec with SVC**.
    You can choose one of them at the dropdown list below and there will be more explanation about it. 
    
    -------
    
    """)
    option = st.selectbox(label="Please choose one of the model below", options=("fasttext", "GloVe"))


col1, col2, col3 = st.columns([10, 8, 10])
with col2:
    st.image('assets/Classiflight.png')

st.write("""
    ## Classification Of Issues regarding Flight Operations Based On Customer Questions 
    """)

user_input = st.text_input("What is the question that you would like to ask?",
                           "What is the flight fare from Malaysia to Bangkok?")
st.button("Predict", on_click=set_stage, args=(1,))


if option == 'fasttext':

    if st.session_state.stage > 0:
        prediction = fasttext_model.predict(user_input)
        class_label = prediction[0][0][9:]
        confidence = round(prediction[1][0] * 100, 2)
        st.write(f"The class label is **{class_label}**")

    st.divider()
    st.write("""
    * Available class: flight, airfare, ground_service, airline, abbreviation, aircraft, flight_time, quantity
    """)

    with st.expander("See what is fasttext"):
        st.image('./assets/fasttext.png')
        st.markdown("""
        
        #### What is fasttext?
        FastText is being developed by the Facebook AI Research (FAIR) Team. FastText is famous for its speed
        while being able to maintain on par performance with deep learning classifiers with just using CPU.
        
        #### How does it works?
        The following image shows the text classifier __fastText__ model architecture.
        """)

        st.image('./assets/fasttext_model.png')

        st.markdown("""
        The figure above shows a simple linear model with rank constraints. The word representation are averaged 
        into a text representation and fed into the linear classifier. The text representation is a hidden variable 
        which can be potentially be reused. This is similar to the CBOW model where the middle word is replaced by a label. 
        
        Then the model uses a softmax function to compute the probability distribution over the predefined classes. The model
        is then trained asynchronously on multiple CPUs using stochastic gradient descent and a linearly decaying learning 
        rate.
        
        #### What can it do?
        FastText can be used in the application of text classification and text representation learning. 
        There are a lot of approaches that can be used in these two application but fasttext can outperform
        them in terms of training time while being able to maintain the accuracy.
        """)

# GloVe explanation
elif option == 'GloVe':

    if st.session_state.stage > 0:

        text_vector = [vectorise(tokenise_sentence(user_input))]
        predicted_label = word2vec_model.predict(text_vector)
        result = label_encoder.inverse_transform(predicted_label)

        st.write(f"The class label is **{result[0][5:]}**")

    st.divider()
    st.write("""
        * Available class: flight, airfare, ground_service, airline, abbreviation, aircraft, flight_time, quantity
        """)
    with st.expander("See what is GloVe"):
        st.markdown(f"""
    
            ------
    
            ### GloVe 
            
            #### What is GloVe?
            GloVe is an unsupervised learning algorithm for obtaining vector representations for words (Pennington, 2014).
            
            #### How does it works?
            It uses the cosine similarity between two words to measure their semantic similarity. For example, words close to frog are frogs, toad, litoria. 
            
            """)

        st.image('assets/cosine_similarity.png')

        st.markdown("""
        However, this simplicity can be problematic. In order to capture the nuance to distinguish man from woman, 
        GloVe used linear substructures and vector difference between two word vectors to capture meaning of the two words. 
        Linear substructures are commonly referred to the way that word vectors are arranged in a vector space.
            """)
        
        st.image('assets/linear_substructures.jpg')
        
        st.markdown("""
        #### What can GloVe do?
        glove embeddings in natural languages processing tasks such as language translation, text classification, and information retrieval. 
        The GloVe algorithm uses a co-occurrence matrix to learn the relationships between words, 
        and it can be trained on large datasets to learn rich and accurate embeddings. 
        
        #### Let's take an easy example:
        
        """)

        st.image('./assets/man_woman.png')

        st.markdown("""
        The underlying concept that distinguishes man from woman, i.e. sex or gender, 
        may be equivalently specified by various other word pairs, such as king and queen.
        To state this observation mathematically, we might expect that the vector differences man - woman and king - queen to be roughly equal.

        ### What we did
        After getting the word embeddings of the sentences using GloVe, we trained it on a Support Vector Classifier, in which the output of it will be the relevant intent labels of the customer. 
        """)
