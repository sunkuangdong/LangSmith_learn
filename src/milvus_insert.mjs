import "dotenv/config";
import { existsSync, readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MilvusClient, DataType, IndexType, MetricType } from "@zilliz/milvus2-sdk-node";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";

const COLLECTION = process.env.MILVUS_COLLECTION ?? 'rag_docs';
const MILVUS_ADDRESS =
  process.env.MILVUS_URI?.replace(/^https?:\/\//, '') ?? 'localhost:19530';

const embeddingDimensions = Number(process.env.OPENAI_EMBEDDING_DIMENSIONS);

const embeddings = new OpenAIEmbeddings({
  apiKey:
    process.env.OPENAI_EMBEDDING_API_KEY ?? process.env.OPENAI_API_KEY,
  model:
    process.env.OPENAI_EMBEDDING_MODEL ??
    'text-embedding-3-small',
  dimensions: Number.isFinite(embeddingDimensions)
    ? embeddingDimensions
    : undefined,
  configuration: {
    baseURL:
      process.env.OPENAI_EMBEDDING_BASE_URL ??
      process.env.OPENAI_BASE_URL ??
      'https://api.openai.com/v1',
  },
});

async function loadChunks(dataDir = "./data") {
    if (!existsSync(dataDir)) {
      throw new Error(`数据目录不存在: ${dataDir}`);
    }
    const files = readdirSync(dataDir).filter((f) => /\.(txt|md)$/i.test(f));
    if (files.length === 0) {
      throw new Error(`目录内无 .txt/.md 文件: ${dataDir}`);
    }
  
    const docs = files.map((f) => ({
      pageContent: readFileSync(join(dataDir, f), "utf-8"),
      metadata: { source: f },
    }));

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });
    return splitter.splitDocuments(docs);
}

const client = new MilvusClient({ address: MILVUS_ADDRESS });

async function main() {
    try {
        console.log("Connecting to Milvus...");
        await client.connectPromise;
        console.log("✓ Connected\n");

        const chunks = await loadChunks();

        if ((await client.hasCollection({ collection_name: COLLECTION })).value) {
            await client.dropCollection({ collection_name: COLLECTION });
            console.log(`Dropped collection: ${COLLECTION}\n`);
        }

        console.log("Generating embeddings...");
        const vectors = await embeddings.embedDocuments(
            chunks.map((c) => c.pageContent),
        );
        const dim = vectors[0].length;

        await client.createCollection({
            collection_name: COLLECTION,
            fields: [
                {
                    name: "langchain_primaryid",
                    data_type: DataType.Int64,
                    is_primary_key: true,
                    autoID: true,
                },
                { name: "langchain_vector", data_type: DataType.FloatVector, dim },
                { name: "langchain_text", data_type: DataType.VarChar, max_length: 8000 },
                { name: "source", data_type: DataType.VarChar, max_length: 256 },
            ],
        });

        console.log("Collection created");

        await client.createIndex({
            collection_name: COLLECTION,
            field_name: "langchain_vector",
            index_type: IndexType.IVF_FLAT,
            metric_type: MetricType.L2,
            params: { nlist: 128 },
        });
        console.log("Index created");

        console.log("\nLoading collection...");
        await client.loadCollection({ collection_name: COLLECTION });
        console.log("Collection loaded");

        console.log("\nInserting...");
        const data = chunks.map((chunk, i) => ({
            langchain_text: chunk.pageContent,
            langchain_vector: vectors[i],
            source: chunk.metadata.source,
        }));

        await client.insert({
            collection_name: COLLECTION,
            data: data,
        });
        console.log("Inserted");
    } catch (error) {
        console.error('Error connecting to Milvus:', error);
        process.exit(1);
    } finally {
        await client.closeConnection();
    }
}

main()

