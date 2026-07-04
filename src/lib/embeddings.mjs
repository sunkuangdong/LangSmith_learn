import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddingDimensions = Number(process.env.OPENAI_EMBEDDING_DIMENSIONS);

export const embeddings = new OpenAIEmbeddings({
  apiKey:
    process.env.OPENAI_EMBEDDING_API_KEY ?? process.env.OPENAI_API_KEY,
  model: process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small",
  dimensions: Number.isFinite(embeddingDimensions)
    ? embeddingDimensions
    : undefined,
  configuration: {
    baseURL:
      process.env.OPENAI_EMBEDDING_BASE_URL ??
      process.env.OPENAI_BASE_URL ??
      "https://api.openai.com/v1",
  },
});
