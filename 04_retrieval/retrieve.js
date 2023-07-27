import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Chroma } from 'langchain/vectorstores/chroma';

let embedding = new OpenAIEmbeddings();

// First run only
let texts = [
    'The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).',
    'A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.',
    'A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.'
];

let metadata = [{id: 1}, {id: 2}, {id: 3}];

// RUN 1ST TIME ONLY
let vectorStore1 = await Chroma.fromTexts(
    texts,
    metadata,
    embedding,
    { collectionName: 'mushroom-collection' }
);


// Lets resuse the persisted store
/*
let vectorStore = await Chroma.fromExistingCollection(
    embedding,
    { collectionName: 'mushroom-collection' }
);
*/
