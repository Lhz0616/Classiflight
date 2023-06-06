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
    option = st.selectbox(label="Please choose one of the model below", options=("fasttext", "Word2Vec"))


col1, col2, col3 = st.columns([10, 8, 10])
with col2:
    st.image('assets/Classiflight.png')

st.write("""
    ## Classification Of Issues regarding Flight Operations Based On Customer Questions 
    """)

user_input = st.text_input("What is the question that you would like to ask?",
                           "What is the flight fare from Malaysia to Bangkok?")
st.button("Predict", on_click=set_stage, args=(1,))

st.write("""
* Available class: flight, airfare, ground_service, airline, abbreviation, aircraft, flight_time, quantity
""")


# fasttext explanation
st.divider()

if option == 'fasttext':

    if st.session_state.stage > 0:
        prediction = fasttext_model.predict(user_input)
        st.write(f"The class label is **{prediction[0][0][9:]}** with a confidence of **{prediction[1][0]}**")

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


# Word2Vec explanation
elif option == 'Word2Vec':

    if st.session_state.stage > 0:

        text_vector = [vectorise(tokenise_sentence(user_input))]
        predicted_label = word2vec_model.predict(text_vector)
        result = label_encoder.inverse_transform(predicted_label)

        st.write(f"The class label is **{result[0][5:]}**")

    with st.expander("See what is Word2Vec"):
        st.markdown(f"""
    
            ------
    
            ### Word2Vec 
            
            #### What is Word2Vec?
            Word2Vec is a two-layer neural net that processes text by "vectorizing" words.
            It changes text into numerical forms.  
            
            #### How does it works?
            Word2Vec uses a trick that we may seen elsewhere in machine learning. 
            Word2Vec uses a simple neural network with a single hidden layer, with weights and biases too. 
            In this approach, we want to reduce the loss function during training and we will take the weights
            as the word embeddings.
            
            The following is the picture of how the model actually is:
            
            """)

        st.image('./assets/skip_gram_net_arch.png')

        st.markdown("""
        First of all, we cannot feed a word as string into a neural network.
        Instead, the word is one-hot-encoded into vectors with the length of the size of vocabulary
        and fill all of them with zeros and only a "1" for the word we want. This will be the input layer. 
        
        Then, the input layer is connected to the hidden layer and the hidden layer is then
        connected to the output layer, which is a softmax classifier. 
        
        The hidden layer weight matrix is what we want, as they are the word vectors that we want. 
        This is how we obtain the Word2Vec vector. 
        
        #### What can Word2Vec do?
        Word2Vec is effective in grouping the vectors of the similar words. If there's a big enough dataset, then
        Word2Vec is able to produce strong estimation about a word's meaning. 
        
        #### Let's take an easy example:
        We want to find the word 'queen' with the words 'king', 'man', 'woman'. If in the case of mathematics,
        we know that if we remove 'man' and add 'woman' into the 'king', then we are able to get 'queen'.
        
        But how does Word2Vec do this?
        
        """)

        st.image('./assets/Word2Vec_example.png')

        st.markdown("""
        As you can see from the photo above, we can see that we will take the coordinates of the different words 
        and do the mathematical operation:
        
        $ 'king' - 'man' + 'woman' = 'queen' $
        
        $ [5, 3] - [2, 1] + [3, 2] = [6, 4] $
        """)



