package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.ollama.{OllamaChatModel, OllamaEmbeddingModel}
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever
import dev.langchain4j.service.{UserMessage, V}
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore
import it.unibo.tools.MathModule

import java.nio.file.Paths
import java.util.HashMap

// 1. RAG-Equipped Biology Research Subagent
trait UnifiedBioResearchAgent:
  @UserMessage(Array("""
    Search the scientific fish biology database and answer this specific question: {{question}}.
    Be accurate and rely solely on the retrieved scientific texts.
  """))
  @Agent(outputKey = "researchResult", description = "Searches the local fish biology research database to answer scientific queries about habitat, spawning, lifecycle, or physical features of fish.")
  def research(@V("question") question: String): String

// 2. Tool-Equipped Scientific Calculator Subagent
trait UnifiedCalculationAgent:
  @UserMessage(Array("""
    Solve this scientific arithmetic calculation: {{problem}}.
    Use the available math tools to guarantee perfect numerical accuracy.
  """))
  @Agent(outputKey = "calcResult", description = "Solves double-precision calculations, conversions, and statistics using mathematics tools (sum, subtract, multiply, divide).")
  def calculate(@V("problem") problem: String): String

// 3. Academic Report Formatter Subagent
trait UnifiedReportAgent:
  @UserMessage(Array("""
    Construct a premium academic report in markdown format.
    Research Findings to include: "{{researchResult}}"
    Calculations performed to include: "{{calcResult}}".
    Format the report professionally with a title, clear sections, a formatted bulleted summary, and a brief discussion on ecological implications.
  """))
  @Agent(outputKey = "reportResult", description = "Formulates professional academic markdown reports based on prior research findings and calculations in the scope.")
  def compileReport(): String


@main
def runUnifiedExample(): Unit =
  println("===============================================")
  println("STAGE 1: RUNNING OFFLINE RAG INGESTION (LONG-TERM MEMORY)")
  println("===============================================")

  // 1. Locate the biology documents folder
  val resourceUrl = Thread.currentThread().getContextClassLoader.getResource("docs")
  val docsPath = if (resourceUrl != null && resourceUrl.getProtocol == "file") {
    Paths.get(resourceUrl.toURI)
  } else {
    Paths.get("src/main/resources/docs")
  }
  println(s"Loading scientific documents from: $docsPath")
  val docs = FileSystemDocumentLoader.loadDocuments(docsPath)

  // 2. Set up Ollama Embedding Model
  val embeddingModel = OllamaEmbeddingModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("ibm/granite-embedding:30m")
    .build()

  val store = new InMemoryEmbeddingStore[TextSegment]()

  // 3. Ingest documents into Vector Store
  println("Embedding and indexing documents...")
  EmbeddingStoreIngestor.builder()
    .embeddingModel(embeddingModel)
    .embeddingStore(store)
    .build()
    .ingest(docs)
  println(s"Ingestion complete. Registered ${store.size()} text segments.")

  // 4. Create retriever
  val retriever = EmbeddingStoreContentRetriever.builder()
    .embeddingStore(store)
    .embeddingModel(embeddingModel)
    .maxResults(3) // Retrieve top 3 relevant chunks
    .build()

  println("\n===============================================")
  println("STAGE 2: INITIALIZING INDIVIDUAL SPECIALIZED AGENTS")
  println("===============================================")

  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val toolModel = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("qwen2.5:latest")
    .logRequests(false)
    .logResponses(false)
    .build()

  // Build BioResearchAgent with RAG content retriever (Long-term Memory)
  println("- Building RAG-equipped BioResearchAgent...")
  val bioResearchAgent = AgenticServices.agentBuilder(classOf[UnifiedBioResearchAgent])
    .chatModel(model)
    .contentRetriever(retriever)
    .build()

  // Build CalculationAgent with Math Tools (Function Calling)
  println("- Building Tool-equipped CalculationAgent...")
  val mathModule = new MathModule()
  val calculationAgent = AgenticServices.agentBuilder(classOf[UnifiedCalculationAgent])
    .chatModel(toolModel)
    .tools(mathModule)
    .build()


  // Build ReportAgent
  println("- Building formatting ReportAgent...")
  val reportAgent = AgenticServices.agentBuilder(classOf[UnifiedReportAgent])
    .chatModel(model)
    .build()

  println("\n===============================================")
  println("STAGE 3: ORCHESTRATING VIA MULTI-AGENT SUPERVISOR")
  println("===============================================")

  val supervisor = AgenticServices.supervisorBuilder()
    .chatModel(model)
    .subAgents(bioResearchAgent, calculationAgent, reportAgent)
    .supervisorContext("""
      You are the Master Scientific Coordinator of a marine biology laboratory.
      Your goal is to answer scientific research queries by orchestrating your specialized assistants:
      - Use BioResearchAgent when asked scientific questions about fish (habitat, spawning, migration, biology).
      - Use CalculationAgent when mathematical computations, multiplication, or division are required.
      - Use ReportAgent when asked to generate a comprehensive markdown report, executive summary, or academic outline using prior findings.
      Ensure you pass accurate, natural queries to each subagent.
      IMPORTANT: If a subagent has already been called and returned a result, DO NOT call it again. Immediately invoke 'done' with the final answer.
    """)
    .maxAgentsInvocations(2)
    .build()

  println("\n===============================================")
  println("STAGE 4: SIMULATING STATEFUL MULTI-TURN RESEARCH")
  println("===============================================")

  // Combined Turn: Retrieve research, perform calculation, and generate report in one go
  val combinedQuery =
    """Find where European eels spawn and describe their reproductive migration.
      |Then, if European eels cover an average of 35 km per day and their total migration distance is 5500 km, calculate how many days they travel (divide 5500 by 35).
      |Finally, construct a premium academic markdown report summarizing the research findings and the calculation result, with a title, clear sections, a formatted bulleted summary, and a brief discussion on ecological implications.
      |Use BioResearchAgent for the research, CalculationAgent for the arithmetic, and ReportAgent to format the final report. If a subagent has already been called and returned a result, do not call it again; reuse the result and finish.""".stripMargin

  println(s"\n[Combined Turn]: Executing unified goal across subagents")

  val finalReport = supervisor.invoke(combinedQuery)
  println("\n--- FINAL COMPREHENSIVE REPORT ---")
  println(finalReport)