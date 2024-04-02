from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import prompts

api_key = "sk-YXYnEtqklrD16nNlIj50z6T3SNuSgvL5P1Vy2q00as2mMNrq"
base_url = "https://api.chatanywhere.tech/v1"
# base_url="https://api.chatanywhere.cn/v1"
os.environ['OPENAI_API_KEY'] = api_key

llm = ChatOpenAI(
    temperature=0.8,
    max_tokens=2048,
    model_name="gpt-3.5-turbo-0125",
    openai_api_base=base_url,
)

memory = ConversationBufferMemory(return_messages=True)

system_message = f"""
You are a senior data scientist tasked with guiding the use of an AutoML 
tool to discover the best XGBoost model configurations for a given binary 
classification dataset. Your role involves understanding the dataset 
characteristics, proposing suitable metrics, hyperparameters, and their 
search spaces, analyzing results, and iterating on configurations. 
"""

# Set up the prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("""{input}""")
])

# Create conversation chain
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=False)

prom = prompts.suggest_metrics()

response = conversation.predict(input=prom)

print(response)
