from langchain import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage , SystemMessage


def get_format_msg(content):
    format_chat_message = [
    SystemMessage(content="You're an Text Converter"),
    HumanMessage(content=f"""
    
format the following text into html , with html tags. Remove all eol , escape , newline characters ,  * , ** ,  \ and / characters :
                 

                 Text :  {content}
""")
]
    return format_chat_message

def get_seo_chat_message(final_summary,  title ):
    chat_message = [
    SystemMessage(content="You're an AI SEO expert writing articles. Create a comprehensive article . Make it 2000 words long"),
    HumanMessage(content=f"""
     use this for artical :  {final_summary}
     End of Text 
Follow these instructions:
Article title : {title}
1. Include different headings #, ## , ###, ####.
3. Insert relevant external links to reputable sources throughout the content. Use at least 5 different sources.
4. Include the target keyword naturally in the first and last sentences of the article.
6. Humanize the output:
   - Use a conversational and natural tone
   - Incorporate analogies and examples where appropriate
   - Add personal anecdotes or experiences (hypothetical if necessary)
   - Use rhetorical questions to engage the reader
   - Vary sentence structure and length for better readability

7. No fluffy AI jargon or generic buzzwords. Keep it neutral and conversational.
8. Create a compelling introduction that hooks the reader.
9. Include bullet points or numbered lists where appropriate for easy readability.
10. Add relevant FAQs at the end of the article.
11. Conclude with a strong summary that reinforces the main points.
""")
]
    return chat_message


seo_template = """
You're an AI SEO expert writing articles to rank my blog. Create a comprehensive article . 

Follow these instructions:

1. Format the headings as HTML (H1, H2, H3, etc.).
3. Insert relevant external links to reputable sources throughout the content. Use at least 5 different sources.
4. Include the target keyword naturally in the first and last sentences of the article.
5. Optimize the content for the given keyword throughout the article.
6. Humanize the output:
   - Use a conversational and natural tone
   - Incorporate analogies and examples where appropriate
   - Add personal anecdotes or experiences (hypothetical if necessary)
   - Use rhetorical questions to engage the reader
   - Vary sentence structure and length for better readability

7. No fluffy AI jargon or generic buzzwords. Keep it neutral and conversational.
8. Create a compelling introduction that hooks the reader.
9. Include bullet points or numbered lists where appropriate for easy readability.
10. Add relevant FAQs at the end of the article.
11. Conclude with a strong summary that reinforces the main points.

use this for artical :  {text}
"""

seo_prompt = PromptTemplate(input_variable = ["text", "word_count"], template = seo_template)

job_template = """
Extract all the key information from this text, which can be used for job posting  : 

{text} , 

"""

job_prompt = PromptTemplate(input_variable = ["text"], template = job_template)

template = """
Please provide the summary of this  : 

{text} , 



"""

prompt = PromptTemplate(input_variable = ["text"], template = template)


final_template = """ 
Provide final summary,   add bullet points, 

Text : {text}
"""

final_prompt = PromptTemplate(input_variables=['text'],template=final_template)
