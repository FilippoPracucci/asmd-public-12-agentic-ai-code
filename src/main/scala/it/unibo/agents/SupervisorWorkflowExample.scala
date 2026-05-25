package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.agentic.supervisor.SupervisorResponseStrategy
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}

trait SuperWriter:
  @UserMessage(Array("Write a very short fun fact about: {{topic}}"))
  @Agent(outputKey = "story", description = "Writes short fun facts about a given topic")
  def write(@V("topic") t: String): String

trait SuperScorer:
  @UserMessage(Array("Rate how interesting this fact is from 0.0 to 1.0: {{story}}"))
  @Agent(outputKey = "score", description = "Evaluates fun facts and provides a score")
  def score(@V("story") s: String): Double

trait TypedSupervisorService:
  def run(@V("query") query: String): String

@main
def runSupervisorWorkflowExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .temperature(0.0)
    .logRequests(false)
    .logResponses(false)
    .build()

  val writer = AgenticServices.agentBuilder(classOf[SuperWriter])
    .chatModel(model)
    .name("SuperWriter")
    .build()

  val scorer = AgenticServices.agentBuilder(classOf[SuperScorer])
    .chatModel(model)
    .name("SuperScorer")
    .build()

  val typedSupervisor = AgenticServices.supervisorBuilder(classOf[TypedSupervisorService])
    .chatModel(model)
    .subAgents(writer, scorer)
    .requestGenerator(scope => scope.readState("query").toString)
    .responseStrategy(SupervisorResponseStrategy.SUMMARY)
    .supervisorContext("""
      Always prefer delegating to the appropriate specialized subagent.
      IMPORTANT:
      If a subagent has already been called and returned a result, DO NOT call it again.
      Immediately invoke 'done' with the final answer.
    """)
    .maxAgentsInvocations(4)
    .build()

  println("Supervisor Workflow configuration built successfully.")

  val query = "Write an interesting fact about octopuses and then score how interesting it is."

  try {
    println(s"\nExecuting typed supervisor workflow with query: '$query'...")
    val typedResult = typedSupervisor.run(query)
    println(s"Result: $typedResult")
  } catch {
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")
  }
