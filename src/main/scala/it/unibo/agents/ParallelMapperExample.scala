package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}
import java.util.concurrent.Executors

trait MapperWriter:
  @UserMessage(Array("Write a one-sentence fun fact about the element: {{topic}}"))
  @Agent(outputKey = "story") def write(@V("topic") t: String): String

trait TypedMapperService:
  def run(@V("topics") topics: java.util.Collection[String]): java.util.Collection[String]

@main
def runParallelMapperExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val writer = AgenticServices.agentBuilder(classOf[MapperWriter]).chatModel(model).build()

  val threadPool = Executors.newFixedThreadPool(3)

  val typedMapper = AgenticServices.parallelMapperBuilder(classOf[TypedMapperService])
    .subAgents(writer)
    .itemsProvider("topics")
    .executor(threadPool)
    .build()

  println("Parallel Mapper Workflow configuration built successfully.")

  val topics = java.util.List.of("Gold", "Helium", "Titanium")

  try
    println("\nExecuting typed parallel mapper workflow...")
    val typedResult = typedMapper.run(topics)
    println(s"Result: $typedResult")
  catch
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")
  finally
    threadPool.shutdown()
