import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import constant
from pydantic import BaseModel, Field
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from skimage import io
import matplotlib.pyplot as plt
from langchain_core.runnables import RunnableLambda

article = constant.ARTICLE

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")


openai_model = 'gpt-4o-mini'

llm = ChatOpenAI(temperature=0.0, model=openai_model, api_key=openai_api_key)

creative_llm = ChatOpenAI(temperature=0.9, model=openai_model, api_key=openai_api_key);

# response = llm.invoke("Say hello in one sentence")
# print(f"\n Reponse: { response.content}")


system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."

)

user_prompt = HumanMessagePromptTemplate.from_template(
        """You are tasked with creating a name for a article.
    The article is here for you to examine {article}

    The name should be based of the context of the article.
    Be creative, but make sure the names are clear, catchy,
    and relevant to the theme of the article.

    Only output the article name, no other explanation or
    text can be provided.""",
        input_variables=["article"]
)

#user_prompt.format(article="TEST STRING")

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# print(first_prompt.format(name= "vasnth", article= "TSRT string"))

chain_one = (
    {
        "article": lambda x: x["article"]
     } 
    | first_prompt
    | creative_llm
    | { "article_title": lambda x: x.content}
)

#asking llm to generate the title 
article_title_msg = chain_one.invoke({"article": article})



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

second_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    second_user_prompt
])

chain_two = (
    {
        "article": lambda x:x['article'],
        "article_title": lambda x: x["article_title"]
    }
    | second_prompt
    | llm
    | { "summary": lambda x: x.content}
)

#asking llm to generate description for artitle
article_description_msg = chain_two.invoke({
    "article": article,
    "article_title": article_title_msg["article_title"]
})

# print(article_title_msg)
# print(article_description_msg)

# Structured output from LangChain
third_user_prompt = HumanMessagePromptTemplate.from_template(
        """You are tasked with creating a new paragraph for the
    article. The article is here for you to examine:

    ---

    {article}

    ---

    Choose one paragraph to review and edit. During your edit
    ensure you provide constructive feedback to the user so they
    can learn where to improve their own writing.""",
        input_variables=["article"]
)

third_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    third_user_prompt
])

#Structing the ouput for langChain

class Paragraph(BaseModel):
    original_paragraph: str = Field(description="The original paragraph")
    edited_paragraph: str = Field(description="The imporoved edited paragraph")
    feedback: str = Field(description=
        "Contructive feedback on the original paragraph"
    )

call_structured_llm = creative_llm.with_structured_output(Paragraph)

chain_three = (
    {
        "article": lambda x:x["article"]
    }
    | third_prompt
    | call_structured_llm
    | {
        "original_paragraph": lambda x: x.original_paragraph,
        "edited_paragraph": lambda x: x.edited_paragraph,
        "feedback": lambda x: x.feedback
    }

)

StructuredOutput = chain_three.invoke({"article": article})

print(StructuredOutput)

#generating image from llms using langChain creating pipeline for it 
image_prompt = PromptTemplate(
    input_variables=["article"],
    template=(
        "Generate a prompt with less then 500 characters to generate an image "
        "based on the following article: {article}"
    )
)

def generate_and_display_image(image_prompt):
    image_url = DallEAPIWrapper().run(image_prompt)
    image