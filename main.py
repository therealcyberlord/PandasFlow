import asyncio
from llama_index.core.tools import FunctionTool
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent, StopEvent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate
import pandas as pd 
from utils import csv_to_metadata_json
from prompts import SYSTEM_PROMPT, HAS_ANSWER_PROMPT, FILTER_COLUMNS_PROMPT, ANSWER_AGENT_SYSTEM_PROMPT
from pydantic import BaseModel
import os 
from dotenv import load_dotenv
from tools import pearson_correlation, filter_column, aggregate
from llama_index.core.agent.workflow import FunctionAgent
from langfuse import Langfuse

load_dotenv()

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
host = os.getenv("LANGFUSE_HOST")

print(secret_key, public_key, host)

# Initialize Langfuse and LlamaIndex instrumentation if environment variables are set
if secret_key and public_key and host:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    langfuse = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=host,
    )
    LlamaIndexInstrumentor().instrument()
    print("Langfuse instrumentation initialized")

class SetupEvent(Event):
    file_name: str
    query: str

class HasAnswerEvent(Event):
    query: str

class DecideColumnEvent(Event):
    query: str

class HasAnswerResponse(BaseModel):
    has_answer: str
    reasoning: str

class FilterColumnsResponse(BaseModel):
    columns: list[str]

pearson_correlation_tool = FunctionTool.from_defaults(pearson_correlation)
filter_column_tool = FunctionTool.from_defaults(filter_column)
aggregate_tool = FunctionTool.from_defaults(aggregate)

class PandasFlow(Workflow):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        df = pd.read_csv(ev.file_name)
        metadata = csv_to_metadata_json(df)
        
        await ctx.store.set("df", df)
        await ctx.store.set("metadata", metadata)
        return StartEvent(query=ev.query, file_name=ev.file_name)
        

    @step
    async def has_answer(self, ctx: Context, ev: StartEvent) -> SetupEvent | HasAnswerEvent | StopEvent:
        query = ev.query
        file_name = ev.file_name

        df = await ctx.store.get("df", default=None)

        if df is None:
            print("Need to load data")
            return SetupEvent(file_name=file_name, query=query)

        metadata = await ctx.store.get("metadata", default=None)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=HAS_ANSWER_PROMPT.format(metadata=metadata, question=query))
        ]
        chat_template = ChatPromptTemplate.from_messages(messages)

        response = await self.llm.astructured_predict(
            prompt=chat_template,
            output_cls=HasAnswerResponse,
            modalities=["text"]
        )

        if not response:
            raise ValueError("Not able to parse HasAnswerResponse")

        if not response.has_answer:
            return StopEvent(result="")

        return HasAnswerEvent(query=query)

    @step
    async def decide_columns(self, ctx: Context, ev: HasAnswerEvent) -> DecideColumnEvent: 
        query = ev.query
        df = await ctx.store.get("df", default=None)
        metadata = await ctx.store.get("metadata", default=None)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=FILTER_COLUMNS_PROMPT.format(metadata=metadata, question=query))
        ]
        chat_template = ChatPromptTemplate.from_messages(messages)

        response = await self.llm.astructured_predict(
            prompt=chat_template,
            output_cls=FilterColumnsResponse,
            modalities=["text"]
        )

        if not response:
            raise ValueError("Not able to parse FilterColumnsResponse")

        filtered_df = df[response.columns]
        await ctx.store.set("df", filtered_df)
        filtered_metadata = csv_to_metadata_json(filtered_df)
        await ctx.store.set("metadata", filtered_metadata)
        return DecideColumnEvent(query=query)
    
    @step
    async def call_answer_agent(self, ctx: Context, ev: DecideColumnEvent) -> StopEvent:  
        query = ev.query
        df = await ctx.store.get("df", default=None)
        metadata = await ctx.store.get("metadata", default=None)

        answer_agent = FunctionAgent(
            llm=self.llm,
            tools=[pearson_correlation_tool, filter_column_tool, aggregate_tool],
            verbose=True,
            allow_parallel_tool_calls=False,
            system_prompt=ANSWER_AGENT_SYSTEM_PROMPT.format(metadata=metadata),
        )
        context = Context(answer_agent)
        await context.store.set("df", df)

        response = await answer_agent.run(user_msg=query, ctx=context)

        return StopEvent(result=response)


async def main():
    w = PandasFlow(timeout=60, verbose=True)
    result = await w.run(query="what is the correlation between BMI and expenses?", file_name="data/insurance.csv")
    print(str(result))


if __name__ == "__main__":
    asyncio.run(main())
