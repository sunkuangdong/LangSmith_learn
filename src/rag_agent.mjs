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
});

const MILVUS_URL =
  process.env.MILVUS_URI?.startsWith("http")
    ? process.env.MILVUS_URI
    : `http://${process.env.MILVUS_URI ?? "localhost:19530"}`;

const vectorStore = await Milvus.fromExistingCollection(embeddings, {
  url: MILVUS_URL,
  collectionName: process.env.MILVUS_COLLECTION ?? "rag_docs",
});

const retriever = vectorStore.asRetriever({ k: 5 });

const retrieve = async (state) => {
  const docs = await retriever.invoke(state.question);
  return { context: docs };
};

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "你是客服助手。仅根据下面「上下文」回答；上下文没有的信息请明确说不知道，不要编造。\n\n上下文：\n{context}",
  ],
  ["human", "{question}"],
]);

const chain = RunnableSequence.from([prompt, llm, new StringOutputParser()]);

const generation = async (state) => {
  const docs = state.context ?? [];
  const contextText = docs.map((c) => c.pageContent).join("\n\n");
  const answer = await chain.invoke({
    context: contextText,
    question: state.question,
  });
  return { answer };
};

const app = new StateGraph(GraphState)
  .addNode("retrieve", retrieve)
  .addNode("generation", generation)
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "generation")
  .addEdge("generation", END)
  .compile();

export async function ask(question) {
  const result = await app.invoke({ question });
  return {
    answer: result.answer,
    context: result.context ?? [],
  };
}
