import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

// Initialize Google GenAI with API key
console.log("Initializing GoogleGenAI...");
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const History = [];

async function transformQuery(question) {
  console.log(`Starting query transformation for question: "${question}"`);

  try {
    // Add user question to history
    console.log("Adding user question to history...");
    History.push({
      role: 'user',
      parts: [{ text: question }]
    });

    console.log("Sending query to Google GenAI for rewriting...");
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
        Only output the rewritten question and nothing else.`,
      },
    });

    // Remove the question from history to keep it clean
    console.log("Removing user question from history...");
    History.pop();

    console.log("Query transformation successful. Rewritten query:", response.text);
    return response.text;
  } catch (error) {
    console.error("Error during query transformation:", error.message);
    throw new Error(`Query transformation failed: ${error.message}`);
  }
}

async function chatting(question) {
  console.log(`Starting chat processing for question: "${question}"`);

  try {
    // Step 1: Transform the query
    console.log("Transforming user query...");
    const queries = await transformQuery(question);
    console.log("Transformed query:", queries);

    // Step 2: Convert question to vector
    console.log("Initializing GoogleGenerativeAIEmbeddings...");
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    console.log("Generating query vector...");
    const queryVector = await embeddings.embedQuery(queries);
    console.log("Query vector generated successfully. Vector length:", queryVector.length);

    // Step 3: Connect to Pinecone and query index
    console.log("Initializing Pinecone client...");
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    console.log("Connecting to Pinecone index:", process.env.PINECONE_INDEX_NAME);
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    console.log("Querying Pinecone index with vector...");
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });
    console.log("Pinecone query successful. Retrieved", searchResults.matches.length, "matches");

    // Step 4: Process search results
    console.log("Extracting context from search results...");
    const context = searchResults.matches
      .map((match, index) => {
        console.log(`Processing match ${index + 1}: Score = ${match.score}`);
        return match.metadata.text;
      })
      .join("\n\n---\n\n");
    console.log("Context created successfully. Context length:", context.length);

    // Step 5: Generate response using Google GenAI
    console.log("Adding transformed query to history...");
    History.push({
      role: 'user',
      parts: [{ text: queries }]
    });

    console.log("Sending query to Google GenAI for final response...");
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `You are a Pakistan History Expert.
        You will be given a context of relevant information and a user question.
        Your task is to answer the user's question based ONLY on the provided context.
        If the answer is not in the context, you must say "I could not find the answer in the provided document."
        Keep your answers clear, concise, and educational.
        
        Context: ${context}`,
      },
    });

    // Step 6: Add model response to history
    console.log("Adding model response to history...");
    History.push({
      role: 'model',
      parts: [{ text: response.text }]
    });

    console.log("\n=== Final Response ===");
    console.log(response.text);
    console.log("=====================\n");

  } catch (error) {
    console.error("Error in chatting function:", error.message);
    console.log("Attempting to continue with next question...");
  }
}

async function main() {
  console.log("Starting main application loop...");
  while (true) {
    try {
      const userProblem = readlineSync.question("Ask me anything--> ");
      console.log(`User input received: "${userProblem}"`);
      
      if (userProblem.toLowerCase() === 'exit') {
        console.log("User requested exit. Terminating application...");
        break;
      }

      await chatting(userProblem);
    } catch (error) {
      console.error("Error in main loop:", error.message);
      console.log("Continuing to next iteration...");
    }
  }
  console.log("Application terminated.");
}

console.log("Starting application...");
main().catch(error => {
  console.error("Fatal error in application:", error.message);
  process.exit(1);
});
