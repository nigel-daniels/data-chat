import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { NotionLoader } from 'langchain/document_loaders/fs/notion';

// Retrieval augmented generation
// In retrieval augmented generation (RAG), an LLM retrieves contextual
// documents from an external dataset as part of its execution.
// This is useful if we want to ask question about specific documents
// (e.g., our PDFs, a set of videos, etc).

// PDF Loader
let loader1 = new PDFLoader('../data/MachineLearning-Lecture01.pdf');
let docs1 = await loader1.load();

console.log('Loaded Document');
console.log('docs length: ' + docs1.length);
console.log('\nfirst doc  : ' + JSON.stringify(docs1[0].pageContent.substring(0, 500)));
console.log('\nmetadata   : ' + JSON.stringify(docs1[0].metadata));


// There is NO YouTube Loader for the JS API

// URL loader
let loader2 = new CheerioWebBaseLoader('https://github.com/basecamp/handbook/blob/master/37signals-is-you.md');
let docs2 = await loader2.load();

console.log('\nLoaded WebPage');
console.log('first doc : ' + JSON.stringify(docs2[0].pageContent.substring(0, 500)));

// Notion Loader
// To use this create an account on notion.so then go here
// and duplicate the page, next export the duplicated pages as 'Markdown & CSV'
// include suppages and add them to subfolders.
let loader3 = new NotionLoader('../data/notion');
let docs3 = await loader3.load();

console.log('\n\nLoaded Notion Directory');
console.log('first doc  : ' + JSON.stringify(docs3[0].pageContent.substring(0, 200)));
console.log('\nmetadata   : ' + JSON.stringify(docs3[0].metadata));

// Note: all of the documents are large so we need to split the documents into
// smaller splits that can be consumed by the LLM
