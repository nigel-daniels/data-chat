import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { RetrievalQAChain, loadQAMapReduceChain, loadQARefineChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { OpenAI } from 'langchain/llms/openai';

let embedding = new OpenAIEmbeddings();
let llm = new OpenAI({temperature: 0.0});
let chatLLM = new ChatOpenAI({temperature: 0.0});

console.log('Load Vector Store\n-----------------');
let vectorStore = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'test-collection' }
);

let collectionCount = await vectorStore.collection.count();
console.log('Test collection: ' + collectionCount);


console.log('\nSimilarity Test\n--------------');
let question1 = 'What are major topics for this class?';

let results1 = await vectorStore.similaritySearch(question1, 3);
console.log('Sim Search: ' + results1.length);


console.log('\nRetrieval QA\n------------');
let qaRetrieverChain = RetrievalQAChain.fromLLM(
    chatLLM,
    vectorStore.asRetriever()
);

let result1 = await qaRetrieverChain.call({'query': question1});
console.log('result: '+ result1.text);


console.log('\nPrompt\n------');
let template = 'Use the following pieces of context to answer the question at ' +
    'the end. If you don\'t know the answer, just say that you don\'t know, ' +
    'don\'t try to make up an answer. Use three sentences maximum. Keep the ' +
    'answer as concise as possible. Always say "thanks for asking!" at the ' +
    'end of the answer.\n{context}\nQuestion: {question}\nHelpful Answer:'

let prompt = PromptTemplate.fromTemplate(template);

let qaPromptChain = RetrievalQAChain.fromLLM(
    chatLLM,
    vectorStore.asRetriever(),
    {
        returnSourceDocuments: true,
        prompt: prompt
    }
);

let question2 = 'Is probability a class topic?';

let result2 = await qaPromptChain.call({query: question2});

console.log('result: ' + result2.text);
for (const source of result2.sourceDocuments) {
    console.log('source: ' + source.metadata.source);
}


console.log('\nRetrieval QA - MR\n----------------');
let mrRetrieverChain = new RetrievalQAChain({
    combineDocumentsChain: loadQAMapReduceChain(chatLLM),
    retriever: vectorStore.asRetriever()
});

let result3 = await mrRetrieverChain.call({query: question2});
console.log('result: ' + result3.text);


console.log('\nRetrieval QA - Refine\n--------------------');
let reRetrieverChain = new RetrievalQAChain({
    combineDocumentsChain: loadQARefineChain(llm),
    retriever: vectorStore.asRetriever()
});

let result4 = await reRetrieverChain.call({query: question2});
console.log('result: ' + result4.output_text);


console.log('\nLimitation - history\n---------------------');
let question3 = 'Is probability a class topic?';

let result5 = await qaRetrieverChain.call({'query': question3});
console.log('result: '+ result5.text);

let question4 = 'why are those prerequesites needed?';

let result6 = await qaRetrieverChain.call({'query': question4});
console.log('\nresult: '+ result6.text);
