import "dotenv/config";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { Milvus } from "@langchain/community/vectorstores/milvus";
import { embeddings } from "./lib/embeddings.mjs";
import { llm } from "./lib/llm.mjs";

const GraphState = Annotation.Root({
    question: Annotation(),
    answer: Annotation(),
    context: Annotation(),
})

const VectorStore = Milvus.fromExistingCollection(embeddings, {
    collection_name: process.env.MILVUS_COLLECTION,
    uri: process.env.MILVUS_URI,
})
const retriever = VectorStore.asRetriever({k:5})

const retrieve = async (state) => {
    await retriever.invoke(state.question)
    return {
        context: results,
    }
}
const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是客服助手。仅根据下面「上下文」回答；上下文没有的信息请明确说不知道，不要编造。\n\n上下文：\n{context}",
    ],
    ["human", "{question}"],
]);
const chain = RunnableSequence.from([prompt, llm, new StringOutputParser()]);

const generation = async (state) => {
    const context = state.context.map((c) => c.pageContent).join("\n")
    const result = await chain.invoke({
        context: context,
        question: state.question,
    })
    return {
        answer: result,
    }
}
const workflow = new StateGraph(GraphState)
    .addNode("retrieve", retrieve)
    .addNode("generation", generation)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generation")
    .addEdge("generation", END)

const compile = workflow.compile()

const ask = async (question) => {
    const result = await compile.invoke({ question: question })
    return result.answer
}
export default { compile, ask }