import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import constant


article = constant.ARTICLE
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")


openai_model = 'gpt-4o-mini'

llm = ChatOpenAI(temperature=0.0, model=openai_model, api_key=openai_api_key);

creative_llm = ChatOpenAI(temperature=0.9, model=openai_model, api_key=openai_api_key)


second_user_prompt = HumanMessagePromptTemplate.from_template(
        """You are tasked with creating a description for
    the article. The article is here for you to examine:

    ---

    {article}

    ---

    Here is the article title '{article_title}'.

    Output the SEO friendly article description. Do not output
    anything other than the description.""",
        input_variables=["article", "article_title"]
)

