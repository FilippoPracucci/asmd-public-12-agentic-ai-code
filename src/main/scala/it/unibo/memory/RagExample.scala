package it.unibo.memory

import dev.langchain4j.data.document.loader.FileSystemDocumentLoader
import dev.langchain4j.data.document.splitter.DocumentSplitters
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.ollama.{OllamaChatModel, OllamaEmbeddingModel}
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever
import dev.langchain4j.rag.query.Query
import dev.langchain4j.service.AiServices
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore

import java.nio.file.Paths


// Automatically chunk, embed, and ingest

@main
def testRag(): Unit =
  // load resources in /docs resource and manage it as path
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
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(true)
    .logResponses(true)
    .build()

  val ingestResult = EmbeddingStoreIngestor.builder()
    .documentSplitter(DocumentSplitters.recursive(150, 15)) // Smaller segment size to break sentences cleanly and avoid embedding the whole document
    .embeddingModel(embeddingModel)
    .embeddingStore(store)
    .build()
    .ingest(docs)

  println(s"Total ingested text segment chunks: ${store.size()}")

  val retriever = EmbeddingStoreContentRetriever.builder()
    .embeddingStore(store)
    .embeddingModel(embeddingModel)
    .maxResults(3) // Retrieve top 3 relevant chunks
    .build()

  val eelResult = retriever.retrieve(Query.from("Where do European eels spawn?"))
  println(eelResult)

  val assistant = AiServices.builder(classOf[Assistant])
    .chatModel(model)
    .contentRetriever(retriever)
    .build()

  println(assistant.chat("Where do European eels spawn?"))


