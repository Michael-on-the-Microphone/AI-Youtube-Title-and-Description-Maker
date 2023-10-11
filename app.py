                                                                                                    #bring in deps
import os
from apikey import apikey

import streamlit as st                                                                              #app framework
from langchain.llms import OpenAI                                                                   #open ai service
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
                                                                                                    #app framework

os.environ['OPENAI_API_KEY'] = apikey


#APPLICATION FRAMEWORK

st.title('yt GPT')                                                                          #app name
prompt = st.text_input('tel subject i give title ?')                                                      #app line

#Prompt Template
title_template = PromptTemplate(
    input_variables= ['topic'],
    template='Write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables= ['topic'],
    template='Write me a youtube video script with the title TITLE: {topic}'
)





#LLM

llm =  OpenAI(temperature=0.9)                                                                 #llm creativity level
title_chain = LLMChain(llm=llm, prompt=title_template,  verbose=True, output_key= 'title')
script_chain = LLMChain(llm=llm, prompt=script_template,  verbose=True, output_key= 'script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)



if prompt:                                                                                     #prompt input and manage
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

