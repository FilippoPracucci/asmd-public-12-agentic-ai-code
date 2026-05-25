package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}
import java.util.concurrent.Executors

trait ParallelExpert:
  @UserMessage(Array("Suggest a great classic movie about: {{topic}}"))
  @Agent(outputKey = "movies") def plan(@V("topic") topic: String): String

trait TypedParallelService:
  def run(@V("topic") topic: String): String

@main
def runParallelExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val expert = AgenticServices.agentBuilder(classOf[ParallelExpert]).chatModel(model).build()

  val threadPool = Executors.newFixedThreadPool(2)

  val typedParallel = AgenticServices.parallelBuilder(classOf[TypedParallelService])
    .subAgents(expert, expert)
    .executor(threadPool)
    .output(scope => {
      val result = scope.readState("movies")
      if (result == null) "No movie suggestions were generated."
      else s"Recommendations aggregated: $result"
    })
    .build()

  println("Parallel Workflow configuration built successfully.")

  try
    println("\nExecuting typed parallel workflow...")
    val typedResult = typedParallel.run("space exploration")
    println(s"Result: $typedResult")
  catch
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")
  finally
    threadPool.shutdown()
