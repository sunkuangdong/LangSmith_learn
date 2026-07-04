import "dotenv/config";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { ask } from "../rag_agent.mjs";
import { ragEvaluators } from "./evaluators.mjs";

const DATASET_NAME = "rag-eval-v1";
const client = new Client({ apiKey: process.env.LANGCHAIN_API_KEY });

async function runRagAgent(inputs) {
    const {answer, context} = await ask(inputs.question);
    return {
        answer,
        context: context.map(d => d.pageContent),
    };
}

async function main() {
    const results = await evaluate(runRagAgent, {
        data: DATASET_NAME,
        evaluators: ragEvaluators,
        client: client,
        experimentPrefix: "rag-eval",
        maxConcurrency: 2,
    });
    
    let count = 0;
    for await (const row of results) {
        count++;
        console.log(`已完成 ${count} 条`);
    }

    const project = process.env.LANGCHAIN_PROJECT ?? "default";
    console.log("✅ 评测完成");
    console.log("实验名:", results.experimentName);
    console.log(
        "指标: rag_groundedness | rag_helpfulness | rag_retrieval_relevance",
    );
    console.log(
        `报告: https://smith.langchain.com/o/default/projects/p/${encodeURIComponent(project)}`,
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
