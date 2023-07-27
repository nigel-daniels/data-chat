import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import nj from 'numjs';

// In loading the docs we deliberately load lecture 01 twice to simulate
// having some bad data
console.log('Loading PDFs\n------------');

let loaders = [
    new PDFLoader('../data/MachineLearning-Lecture01.pdf'),
    new PDFLoader('../data/MachineLearning-Lecture01.pdf'),
    new PDFLoader('../data/MachineLearning-Lecture02.pdf'),
    new PDFLoader('../data/MachineLearning-Lecture03.pdf'),
];

let docs = [];

for (const loader of loaders) {
    docs = docs.concat(await loader.load());
}

console.log('Docs total: ' + docs.length);

// Now we need to split the documents
console.log('\nSplitting documents\n-------------------');

let rSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500,
  chunkOverlap: 150
});

let splits = await rSplitter.splitDocuments(docs);

console.log('Split total: ' + splits.length);

// Now lets use the embeddins
// First lets do some basic tests to see what's happening
console.log('\nEmbedding Test\n--------------');
let embedding = new OpenAIEmbeddings();

let text1 = 'i like dogs';
let text2 = 'i like canines';
let text3 = 'the weather is ugly outside';

let embed1 = await embedding.embedQuery(text1);
let embed2 = await embedding.embedQuery(text2);
let embed3 = await embedding.embedQuery(text3);

console.log('E1 to E2 similarity: ' + nj.dot(embed1, embed2));
console.log('E1 to E3 similarity: ' + nj.dot(embed1, embed3));
console.log('E2 to E3 similarity: ' + nj.dot(embed2, embed3));

// Now lets work with some actual documents
console.log('\nEmbedding Docs\n--------------');

// To use chroma follw the LangChain JS instructions, data is persisted in the chroma directory
/*
let vectorStore = await Chroma.fromDocuments(
    splits,
    embedding,
    { collectionName: 'test-collection' }
);
*/

// Lets resuse the persisted store
let vectorStore = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'test-collection' }
);

console.log('Vector store ready.');

let collectionCount = await vectorStore.collection.count();
console.log('\nVector collection: ' + collectionCount);

let query1 = 'is there an email i can ask for help';

let results1 = await vectorStore.similaritySearch(query1, 3);
console.log('result1 len: ' + results1.length);
console.log('1st chunk  : ' + results1[0].pageContent);

// NOTE with the JS library it auto persists in the chrome directory

// There are some fail modes
let query2 = 'what did they say about matlab?';

let results2 = await vectorStore.similaritySearch(query2, 5);
console.log('\nFail mode 1 - duplicates');
console.log('chunk 1: ' + results2[0].pageContent);
console.log('chunk 2: ' + results2[1].pageContent);

let query3 = 'what did they say about regression in the third lecture?';

let results3 = await vectorStore.similaritySearch(query3, 5);

console.log('\nFail mode 2 - out of scope');

for (const result of results3) {
    console.log('source: ' + result.metadata.source);
}

console.log('chunk 3: ' + results3[4].pageContent);
