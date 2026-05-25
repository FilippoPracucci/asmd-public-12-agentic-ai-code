package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader
import dev.langchain4j.data.document.splitter.DocumentSplitters
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.ollama.{OllamaChatModel, OllamaEmbeddingModel}
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever
import dev.langchain4j.service.{UserMessage, V}
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore
import it.unibo.tools.MathModule

import java.nio.file.Paths

// 1. RAG-equipped Research Agent
trait UnifiedResearchAgent:
  @UserMessage(Array("Search the fish database and find: {{query}}"))
  @Agent(outputKey = "research", description = "Searches the local fish biology research database to find facts")
  def search(@V("query") query: String): String

// 2. Tool-equipped Calculation Agent
trait UnifiedCalculationAgent:
  @UserMessage(Array("Solve this calculation precisely: {{problem}}"))
  @Agent(outputKey = "calculation", description = "Performs arithmetic calculations using math tools")
  def calculate(@V("problem") problem: String): String

// 3. Document Formatting Agent
trait UnifiedReportAgent:
  @UserMessage(Array("""
    Format a concise research summary report in markdown.
    Research Findings: {{research}}
    Calculations: {{calculation}}
  """))
  @Agent(outputKey = "report", description = "Formats a concise research summary in markdown")
  def formatReport(): String


@main
def runUnifiedExample(): Unit =
  println("===============================================")
  println("STAGE 1: RUNNING OFFLINE RAG INGESTION")
  println("===============================================")

  val resourceUrl = Thread.currentThread().getContextClassLoader.getResource("docs")
  val docsPath = if (resourceUrl != null && resourceUrl.getProtocol == "file") {
    Paths.get(resourceUrl.toURI)
  } else {
    Paths.get("src/main/resources/docs")
  }
  val docs = FileSystemDocumentLoader.loadDocuments(docsPath)
  val store = new InMemoryEmbeddingStore[TextSegment]()
  val embeddingModel = OllamaEmbeddingModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("ibm/granite-embedding:30m")
    .build()

  EmbeddingStoreIngestor.builder()
    .documentSplitter(DocumentSplitters.recursive(150, 15)) // Smaller segment size to break sentences cleanly and avoid embedding the whole document
    .embeddingModel(embeddingModel)
    .embeddingStore(store)
    .build()
    .ingest(docs)

  val retriever = EmbeddingStoreContentRetriever.builder()
    .embeddingStore(store)
    .embeddingModel(embeddingModel)
    .maxResults(3)
    .build()

  println("\n===============================================")
  println("STAGE 2: INITIALIZING INDIVIDUAL SPECIALIZED AGENTS")
  println("===============================================")

  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .temperature(0.0) // Set temperature to 0.0 for deterministic tool usage and routing
    .logRequests(false)
    .logResponses(false)
    .build()

  // Build the specialized agents with explicit names matching their supervisor context roles
  val researchAgent = AgenticServices.agentBuilder(classOf[UnifiedResearchAgent])
    .chatModel(model)
    .contentRetriever(retriever) // Equipped with RAG (Long-term memory)
    .name("UnifiedResearchAgent")
    .build()

  val calculationAgent = AgenticServices.agentBuilder(classOf[UnifiedCalculationAgent])
    .chatModel(model)
    .tools(new MathModule()) // Equipped with Tools (Function calling)
    .name("UnifiedCalculationAgent")
    .build()

  val reportAgent = AgenticServices.agentBuilder(classOf[UnifiedReportAgent])
    .chatModel(model)
    .name("UnifiedReportAgent")
    .build()

  println("\n===============================================")
  println("STAGE 3: ORCHESTRATING VIA MULTI-AGENT SUPERVISOR")
  println("===============================================")

  // Build the Supervisor agent that acts as a central coordinator
  val supervisor = AgenticServices.supervisorBuilder()
    .chatModel(model)
    .subAgents(researchAgent, calculationAgent, reportAgent)
    .supervisorContext("""
      You are an expert scientific assistant coordinator.
      Your goal is to coordinate specialized subagents to answer the user's query:
      1. First, invoke `UnifiedResearchAgent` to retrieve the spawning place and migration distance of the European eel.
      2. Second, once you have the migration distance, invoke `UnifiedCalculationAgent` to calculate the travel days (distance / 35 km/day).
      3. Third, once both research and calculation are completed, invoke `UnifiedReportAgent` to format and compile the final markdown summary report.
      4. Finally, once `UnifiedReportAgent` returns the report, immediately finish by calling the `done` tool with the compiled report.
      
      CRITICAL: You must execute these steps sequentially. Do NOT skip the calculation step or the report formatting step.
    """)
    .maxAgentsInvocations(4)
    .build()

  val query = "Create a brief summary report for the European eel migration: find its spawning place and total distance as well as their velocity, then calculate travel days if they swim 35 km/day on average, and compile the final report."
  println(s"Sending query: '$query'")

  val finalReport = supervisor.invoke(query)
  println("\n--- FINAL REPORT ---")
  println(finalReport)