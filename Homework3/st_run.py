import os
import streamlit as st
from serpapi import GoogleSearch
from langchain.tools import StructuredTool
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI


SERPAPI_API_KEY = ""
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["OPENAI_API_KEY"] = ""


def search_tool(query):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY
    })
    results = search.get_dict()
    formatted_results = [result['title'] + ": " + result['snippet'] for result in results.get('organic_results', [])]
    return formatted_results


class ComparisonInput(BaseModel):
    items: List[str]
    category: str

class ComparisonTool:
    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

    def __call__(self, items: List[str], category: str) -> str:
        if not items or not category:
            return "Invalid inputs. Please provide items and a category for comparison."
        
        prompt = f"Compare the following items based on '{category}': {', '.join(items)}."
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return response.content

comparison_tool = StructuredTool.from_function(
    func=ComparisonTool(api_key=os.getenv("OPENAI_API_KEY")).__call__,
    name="comparison_tool",
    description="Compares items based on a given category using GPT",
    input_schema=ComparisonInput
)


class AnalysisTool:
    def __init__(self, api_key):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

    def analyze(self, text: str) -> str:
        prompt = f"Analyze and provide insights for the following text:\n{text}"
        response = self.model.invoke([{"role": "user", "content": prompt}])
        return response.content

analysis_tool = AnalysisTool(api_key=os.getenv("OPENAI_API_KEY"))


class ReActAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.tools = {
            "search": search_tool,
            "compare": comparison_tool,
            "analyze": analysis_tool.analyze
        }

    def respond(self, user_query: str) -> str:
        search_keywords = ["where", "what", "who", "when", "information"]
        comparison_keywords = ["compare", "comparison", "difference", "better", "vs"]
        analysis_keywords = ["analyze", "insight", "summary"]

        if any(keyword in user_query.lower() for keyword in search_keywords):
            result = self.tools["search"](user_query)
        elif any(keyword in user_query.lower() for keyword in comparison_keywords):
            # Parse items and category directly from the query content if needed
            result = self.tools["compare"].invoke({"items": ["product1", "product2"], "category": "price"})
        elif any(keyword in user_query.lower() for keyword in analysis_keywords):
            result = self.tools["analyze"](user_query)
        else:
            result = "I was unable to determine the correct tool."

        return result

react_agent = ReActAgent()


st.title("ReAct Agent Interactive Interface")
st.sidebar.header("Agent Configuration")
model_type = st.sidebar.selectbox("Model Type", ["gpt-3.5-turbo", "gpt-4"])
response_length = st.sidebar.slider("Max Response Length", 50, 300, 150)

st.subheader("Query the ReAct Agent")
query_type = st.selectbox(
    "Select Query Type",
    ["General Query", "Search", "Compare Products", "Analyze Text"]
)

query_content = st.text_input("Enter your query:")
if query_type == "Compare Products":
    items = st.text_area("Enter items to compare (comma-separated):").split(",")
    category = st.text_input("Enter comparison category (e.g., 'price', 'quality')")
elif query_type == "Analyze Text":
    analysis_text = st.text_area("Enter text to analyze:")


if st.button("Submit"):
    if query_type == "General Query":
        response = react_agent.respond(query_content)
    elif query_type == "Search":
        response = react_agent.tools["search"](query_content)
    elif query_type == "Compare Products":
        response = react_agent.tools["compare"].invoke({"items": items, "category": category})
    elif query_type == "Analyze Text":
        response = react_agent.tools["analyze"](analysis_text)
    else:
        response = "Invalid query type selected."
    st.subheader("Agent Response")
    st.write(response)


st.sidebar.subheader("Additional Options")
if st.sidebar.checkbox("Show Raw Response"):
    st.json(response)
