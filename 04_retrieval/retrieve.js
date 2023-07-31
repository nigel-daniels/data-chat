import { OpenAI } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import { AttributeInfo } from 'langchain/schema/query_constructor';
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { ChromaTranslator } from 'langchain/retrievers/self_query/chroma';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';

let embedding = new OpenAIEmbeddings();

console.log('Load Vector Store\n-----------------');
let vectorStore1 = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'test-collection' }
);

let collectionCount1 = await vectorStore1.collection.count();
console.log('Test collection: ' + collectionCount1);


console.log('\nLoad Text\n---------');
// FIRST RUN ONLY (then use the commented code below and comment out this section above it)
let texts = [
    'The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).',
    'A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.',
    'A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.'
];

let metadata = [{id: 1}, {id: 2}, {id: 3}];

// Now lets create a text DB
let vectorStore2 = await Chroma.fromTexts(
    texts,
    metadata,
    embedding,
	{ collectionName: 'mushroom-collection' }
);
/*
let vectorStore2 = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'mushroom-collection' }
);
*/
let question1 = 'Tell me about all-white mushrooms with large fruiting bodies';

let results1 = await vectorStore2.similaritySearch(question1, 2);
console.log('Similarity Search: ' + JSON.stringify(results1));

// NOTE: At the time of writing the MMR Search is not implemented in the JS library

console.log('\nLectures MMR\n--------------');
let question2 = 'what did they say about matlab?';

let results2 = await vectorStore1.similaritySearch(question2, 3);
console.log('Sim Search [0]: ' + JSON.stringify(results2[0].pageContent.substring(0, 100)));
console.log('Sim Search [1]: ' + JSON.stringify(results2[1].pageContent.substring(0, 100)));


// NOTE: At the time of writing the MMR Search is not implemented in the JS library


console.log('\nSpecifity Metadata\n------------------');
let question3 = 'what did they say about regression in the third lecture?';

let results4 = await vectorStore1.similaritySearch(question2, 3, {'source':'../data/MachineLearning-Lecture03.pdf'});
for (const result of results4) {
    console.log('source: ' + result.metadata.source);
}

console.log('\nSpecifity Self-query\n--------------------');
let llm = new OpenAI({temperature: 0.0});

let documentContents = 'Lecture notes';

let attributeInfo = [
    new AttributeInfo(
        'source',
        'The lecture the chunk is from, should be one of `../data/MachineLearning-Lecture01.pdf`, `../data/MachineLearning-Lecture02.pdf`, or `../data/MachineLearning-Lecture03.pdf`',
        'string',
    ),
    new AttributeInfo(
        'page',
        'The page from the lecture',
        'integer',
    )
];

let retriever1 = SelfQueryRetriever.fromLLM({
	llm,
	vectorStore: vectorStore1,
	documentContents,
	attributeInfo,
	structuredQueryTranslator: new ChromaTranslator(),
	verbose: true
});
console.log('Created retriever');

let results5 = await retriever1.getRelevantDocuments(question3);
for (const result of results4) {
    console.log('source: ' + result.metadata.source);
}

console.log('\nCompression\n-----------');
let compressor = LLMChainExtractor.fromLLM(llm);

let retriever2 = new ContextualCompressionRetriever({
	baseCompressor: compressor,
	baseRetriever: vectorStore1.asRetriever()
});

let results6 = await retriever2.getRelevantDocuments(question2);

console.log(printDocs(results6));

console.log('\nCompression + MMR\n-----------------');
// NOTE: At the time of writing the MMR Search is not implemented in the JS library


function printDocs(docs) {
	const separator = '-'.repeat(80);
	const formattedDocs = docs.map((doc, i) => `Document ${i + 1}:\n\n${doc.pageContent}`).join(`\n${separator}\n`);

	return formattedDocs;
}
