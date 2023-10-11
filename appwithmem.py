                                                                                                    #bring in deps
import os
from apikey import apikey

import streamlit as st                                                                              #app framework
from langchain.llms import OpenAI                                                                   #open ai service
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


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
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

#memory

title_memory = ConversationBufferMemory(input_key="topic", memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key="title", memory_key='chat_history')


#LLM

llm =  OpenAI(temperature=0.9)                                                                 #llm creativity level
title_chain = LLMChain(llm=llm, prompt=title_template,  verbose=True, output_key= 'title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template,  verbose=True, output_key= 'script', memory=script_memory)

wiki = WikipediaAPIWrapper()

#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)



if prompt:                                                                                     #prompt input and manage
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    #response = sequential_chain({'topic':prompt})
    script=script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)
    
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia History'):
        st.info(wiki_research)

