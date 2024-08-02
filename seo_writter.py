import os
from dotenv import load_dotenv
load_dotenv()
import markdown
import validators
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import prompts


headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "DNT": "1",
               "Connection": "close", "Upgrade-Insecure-Requests": "1"}


def get_document(url):
    if validators.url(url):
        if 'youtube' in url or 'youtu.be' in url :
            loader = YoutubeLoader.from_youtube_url(url)
        else : 
            #loader = UnstructuredURLLoader(urls=[url], ssl_verify= False, headers = headers)
            loader = WebBaseLoader(web_path=url)
        return loader.load()
    else :
        

        print('Invalid URL')



def document_splitter(document , chunk_size = 5000, chunk_overlap = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(document)


groq_api_key = os.getenv("GROQ_API_KEY")



# groq_api_key = st.secrets["GROQ_API_KEY"]
# os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
# os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGCHAIN_PROJECT"]
os.environ['LANGCHAIN_TRACING_V2'] = "true"

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")






# Streamlit app UI
def app_ui():
    st.title("TanSeo Writter")
    
    
    st.subheader("Enter URL to generate SEO Optimized post")
    url = st.text_input("Enter URL")
    
    
    if st.button("Generate Summary"):
        with st.spinner('Fetching Post...'):
            document = get_document(url)
            splitted_docs = document_splitter(document)
            st.success("Post Fetched.")
        with st.spinner('Reading Post and generating summary...'):
            
            map_reduce_chain = load_summarize_chain(llm=llm , chain_type='map_reduce', map_prompt= prompts.prompt , combine_prompt = prompts.final_prompt )
            final_summary = map_reduce_chain.run(splitted_docs)
            st.success("Post summary generated.")

        with st.spinner('Generating New Post...'):
            
            title = document[0].metadata.get('title', "")
            output = llm(prompts.get_seo_chat_message(final_summary,title ))
            st.success("Post Generated")
        with st.spinner('Formating New Post...'):
            formatted_output = markdown.markdown(output.content)

            formatted_output = llm(prompts.get_format_msg(formatted_output) )
            
            st.success(formatted_output.content)

        

if __name__ == "__main__":
    app_ui()


