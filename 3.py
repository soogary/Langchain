#Load API keys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#Langchain Tools - Search with DuckDuckGo and Wikipedia
#pip install duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
output = search.invoke('Where was Freddie Mercury born?')
print(output)


#search.name
#search.description



from langchain.tools import DuckDuckGoSearchResults
search = DuckDuckGoSearchResults()
output = search.run('Freddie Mercury and Queen')
print(output)
#contrls the output to snippet, title of the request, link
# also use DuckDuckGoSearchAPIWrapper

#output in a user friendly format

import re 
pattern = r'snippet: (.*?), title: (.*?), link: (.*?)\],'
matches = re.finall(pattern, output, re.DOTALL)

for snippet, title, link in matches:
    print(f'Snippet: {snippet}\nTitle: {title}\nLink: {link}\n')
    print('-' * 50)


#Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

#initialize tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki.invoke({'query': 'llamaindex'})

wiki.invoke('Google Gemini')