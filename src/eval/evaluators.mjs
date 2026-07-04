import {
    createLLMAsJudge,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
} from "openevals";
import {llm as judge} from "../lib/llm.mjs";
import { embeddings } from "../lib/embeddings.mjs";

const ragGroundednessJudge = createLLMAsJudge({
    judge,
    prompt: RAG_GROUNDEDNESS_PROMPT,
    feedbackKey: "rag_groundedness",
    continuous: true,
});

const ragHelpfulnessJudge = createLLMAsJudge({
    judge,
    prompt: RAG_HELPFULNESS_PROMPT,
    feedbackKey: "rag_helpfulness",
    continuous: true,
});

const ragRetrievalRelevanceJudge = createLLMAsJudge({
    judge,
    prompt: RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedbackKey: "rag_retrieval_relevance",
    continuous: true,
});

function toDocuments(context) {
    return (context ?? []).map((d) =>
      typeof d === "string" ? d : d.pageContent
    );
}

export async function ragRetrievalRelevanceEvaluator({inputs, outputs}) {
    return ragRetrievalRelevanceJudge({
        inputs,
        context: {
            documents: toDocuments(outputs.context),
        },
    })
}

export async function ragGroundednessEvaluator({inputs, outputs}) {
    return ragGroundednessJudge({
        context: { documents : toDocuments(outputs.context) },
        outputs: { answer: outputs.answer },
    })
}

export async function ragHelpfulnessEvaluator({inputs, outputs}) {
    return ragHelpfulnessJudge({
        inputs,
        outputs: { answer: outputs.answer },
    })
}


export const ragEvaluators = [
    ragGroundednessEvaluator,
    ragHelpfulnessEvaluator,
    ragRetrievalRelevanceEvaluator,
];

