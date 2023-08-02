import { OpenAI } from 'langchain/llms/openai';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import { RetrievalQAChain, ConversationalRetrievalQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { BufferMemory } from 'langchain/memory';


let llm = new OpenAI({temperature: 0.0});
let chatLLM = new ChatOpenAI({temperature: 0.0});
let embedding = new OpenAIEmbeddings();

let vectorStore = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'test-collection' }
);


console.log('\nSimilarity Test\n---------------');
let question1 = 'What are major topics for this class?';

let results1 = await vectorStore.similaritySearch(question1, 3);
console.log('results: ' + results1.length);


console.log('\nChat Test\n---------');
let result1 = await chatLLM.predict('Hello world!');

console.log('result: ' + result1);


console.log('\nQA Chat Test\n------------');
let template = 'Use the following pieces of context to answer the question ' +
    'at the end. If you don\'t know the answer, just say that you don\'t ' +
    'know, don\'t try to make up an answer. Use three sentences maximum. ' +
    'Keep the answer as concise as possible. Always say "thanks for asking!" ' +
    'at the end of the answer.\n{context}\nQuestion: {question}\nHelpful Answer:';

let prompt = new PromptTemplate({
    inputVariables: ['context', 'question'],
    template: template
});

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


console.log('\nMemory Test\n-----------');
let memory = new BufferMemory({
    memoryKey: 'chat_history',
    returnMessages: true
});

let qaMemoryChain = ConversationalRetrievalQAChain.fromLLM(
    chatLLM,
    vectorStore.asRetriever(),
    {memory: memory}
);

let question3 = 'Is probability a class topic?';

let result3 = await qaMemoryChain.call({question: question3});
console.log('result: '+ result3.text);

let question4 = 'why are those prerequesites needed?';

let result4 = await qaMemoryChain.call({question: question4});
console.log('\nresult: '+ result4.text);
