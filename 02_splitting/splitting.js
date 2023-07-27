import { Document } from 'langchain/document';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { TokenTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { NotionLoader } from 'langchain/document_loaders/fs/notion';

// Let's start with basic examples to become familiar with the concepts
console.log('Basic Examples\n--------------');
let chunkSize = 26;
let chunkOverlap = 4;

let rSplitter1 = new RecursiveCharacterTextSplitter({
  chunkSize: chunkSize,
  chunkOverlap: chunkOverlap
});

let cSplitter1  = new CharacterTextSplitter({
	chunkSize: chunkSize,
    chunkOverlap: chunkOverlap
});

// This is 26 chars long so no plitting happens
let text1 = 'abcdefghijklmnopqrstuvwxyz';

let split1 = await rSplitter1.splitText(text1);
console.log('split 01: ' + split1);

// Now lets try > 26 chars, this shows the 26 chars in chunk 1
// with 4 overlap in chunk2
let text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg';

let split2 = await rSplitter1.splitText(text2);
console.log('\nsplit 02: ' + split2);

// Now lets try a more complex example
let text3 = 'a b c d e f g h i j k l m n o p q r s t u v w x y z';

// This example creates 3 chunks
let split3 = await rSplitter1.splitText(text3);
console.log('\nsplit 03: ' + split3);

// With the char text splitter were back to one chunk as the default
// separator is '\n'
let split4 = await cSplitter1.splitText(text3);
console.log('\nsplit 04: ' + split4);

// If we define the separator as ' ' then the splitting shoukld be 3
let cSplitter2  = new CharacterTextSplitter({
	chunkSize: chunkSize,
    chunkOverlap: chunkOverlap,
	separator: ' '
});

let split5 = await cSplitter2.splitText(text3);
console.log('\nsplit 05: ' + split5);


// Now that weve looked at some basic functions let's look at some more
// realistic examples
console.log('\n\nRealistic Examples\n------------------');
let longText1 = 'When writing documents, writers will use document structure to ' +
	'group content. This can convey to the reader, which idea\'s are related. ' +
	'For example, closely related ideas are in sentances. Similar ideas are ' +
	'in paragraphs. Paragraphs form a document. \n\nParagraphs are often ' +
	'delimited with a carriage return or two carriage returns. Carriage ' +
	'returns are the "backslash n" you see embedded in this string. ' +
	'Sentences have a period at the end, but also, have a space and words.' +
	'are separated by space.';

let cSplitter3  = new CharacterTextSplitter({
	chunkSize: 450,
    chunkOverlap: 0,
	separator: ' '
});

let rSplitter2 = new RecursiveCharacterTextSplitter({
  chunkSize: 450,
  chunkOverlap: 0,
  separators: ['\n\n', '\n', ' ', ''] // These are thed efaults and are in order of presidence
});

console.log('longText length: ' + longText1.length);

let split6 = await cSplitter3.splitText(longText1);
console.log('\nsplit 06: ' + split6);

let split7 = await rSplitter2.splitText(longText1);
console.log('\nsplit 07: ' + split7);

// We can add in full stop detection to get sentences we use the regex look
// behind to ensure the '\. ' stays with it's sentence
let rSplitter3 = new RecursiveCharacterTextSplitter({
  chunkSize: 150,
  chunkOverlap: 0,
  separators: ['\n\n', '\n', '(?<=\. )', ' ', ''] // NB the regex is not working??? but '\. ' is
});

let split8 = await rSplitter3.splitText(longText1);
console.log('\nsplit 08: ' + split8);

// Now lets try with document loaders
console.log('\n\nDocument Examples\n-----------------');

let cSplitter4  = new CharacterTextSplitter({
	separator: '\n',
	chunkSize: 1000,
    chunkOverlap: 150,
	lengthFunction: (text) => text.length
});

let loader1 = new PDFLoader('../data/MachineLearning-Lecture01.pdf');
let docs1 = await loader1.load();
console.log('PDF Loaded');

let split9 = await cSplitter4.splitDocuments(docs1);
console.log('docs length  : ' + docs1.length);
console.log('splits length: ' + split9.length);

let loader2 = new NotionLoader('../data/notion');
let docs2 = await loader2.load();
console.log('\nNotion Loaded');

let split10 = await cSplitter4.splitDocuments(docs2);
console.log('docs length  : ' + docs2.length);
console.log('splits length: ' + split10.length);

// Token splitting is useful as LLMs restrict the tokens we can send/receive
console.log('\n\nToken Examples\n--------------');

let tSplitter1 = new TokenTextSplitter({
	chunkSize: 1,
	chunkOverlap: 0
});

let text4 = 'foo bar bazzyfoo';

let split11 = await tSplitter1.splitText(text4);
console.log('split 09: ' + split11);

console.log('\nPDF token split');
let tSplitter2 = new TokenTextSplitter({
	chunkSize: 10,
	chunkOverlap: 0
});

let split12 = await tSplitter1.splitDocuments(docs1);
console.log('first split : ' + JSON.stringify(split12[0]));
console.log('\n1st metadata: ' + JSON.stringify(split12[0].metadata));

// Context aware splitting NOTE: JS does not have an MD splitter but uses
// the RecursiveCharacterTextSplitter.fromLanguage function
// However this does not add contextually rich metadata
console.log('\n\nContext Aware Examples\n----------------------');
let markdown1 = '# Title\n\n ## Chapter 1\n\n Hi this is Jim\n\n ' +
	'Hi this is Joe\n\n ### Section \n\n Hi this is Lance \n\n ' +
	'## Chapter 2\n\n Hi this is Molly';

const lSplitter1 = RecursiveCharacterTextSplitter.fromLanguage('markdown', {
  chunkSize: 32,
  chunkOverlap: 0
});

let split13 = await lSplitter1.createDocuments([markdown1]);
console.log('split 10, doc 0: ' + JSON.stringify(split13[0]));
console.log('split 10, doc 1: ' + JSON.stringify(split13[1]));
