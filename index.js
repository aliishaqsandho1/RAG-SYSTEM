import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

// Validate environment variables
console.log("Validating environment variables...");
if (!process.env.GEMINI_API_KEY || !process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX_NAME) {
  console.error("Missing required environment variables: GEMINI_API_KEY, PINECONE_API_KEY, or PINECONE_INDEX_NAME");
  process.exit(1);
}
console.log("Environment variables validated successfully.");

async function indexDocument() {
  console.log("Starting PDF indexing process...");
  const PDF_PATH = './pakistan.pdf';

  try {
    // Step 1: Load the PDF
    console.log(`Loading PDF from path: ${PDF_PATH}`);
    const pdfLoader = new PDFLoader(PDF_PATH);
    let rawDocs;
    try {
      rawDocs = await pdfLoader.load();
      console.log(`PDF loaded successfully. Number of pages/documents: ${rawDocs.length}`);
    } catch (error) {
      console.error(`Error loading PDF: ${error.message}`);
      throw new Error(`Failed to load PDF: ${error.message}`);
    }

    // Step 2: Chunk the documents
    console.log("Initializing text splitter for chunking...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    let chunkedDocs;
    try {
      console.log("Splitting documents into chunks...");
      chunkedDocs = await textSplitter.splitDocuments(rawDocs);
      console.log(`Chunking completed. Total chunks created: ${chunkedDocs.length}`);
      // Log sample chunk for debugging
      if (chunkedDocs.length > 0) {
        console.log("Sample chunk (first 100 characters):", chunkedDocs[0].pageContent.substring(0, 100));
      }
    } catch (error) {
      console.error(`Error during document chunking: ${error.message}`);
      throw new Error(`Chunking failed: ${error.message}`);
    }

    // Step 3: Configure embedding model
    console.log("Initializing GoogleGenerativeAIEmbeddings...");
    let embeddings;
    try {
      embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
      });
      console.log("Embedding model configured successfully.");
    } catch (error) {
      console.error(`Error initializing embedding model: ${error.message}`);
      throw new Error(`Embedding model initialization failed: ${error.message}`);
    }

    // Step 4: Configure Pinecone
    console.log("Initializing Pinecone client...");
    let pinecone;
    let pineconeIndex;
    try {
      pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });
      console.log("Pinecone client initialized.");
      console.log(`Connecting to Pinecone index: ${process.env.PINECONE_INDEX_NAME}`);
      pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
      console.log("Pinecone index connected successfully.");
    } catch (error) {
      console.error(`Error initializing Pinecone: ${error.message}`);
      throw new Error(`Pinecone initialization failed: ${error.message}`);
    }

    // Step 5: Store chunks in Pinecone
    console.log("Storing chunks in Pinecone with embeddings...");
    try {
      await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
      });
      console.log(`Data stored successfully in Pinecone. Total documents processed: ${chunkedDocs.length}`);
    } catch (error) {
      console.error(`Error storing data in Pinecone: ${error.message}`);
      throw new Error(`Failed to store data in Pinecone: ${error.message}`);
    }

    console.log("\n=== Indexing Process Completed Successfully ===\n");
  } catch (error) {
    console.error("Fatal error in indexDocument:", error.message);
    throw error; // Rethrow to be caught in the main execution
  }
}

console.log("Starting application...");
indexDocument().catch(error => {
  console.error("Application failed:", error.message);
  process.exit(1);
});