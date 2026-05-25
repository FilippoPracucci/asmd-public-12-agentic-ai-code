package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}

trait CondWriter:
  @UserMessage(Array("Write a poem about the topic: {{topic}}"))
  @Agent(outputKey = "story") def write(@V("topic") t: String): String

trait CondScorer:
  @UserMessage(Array("Rate this text's creative tone from 0.0 to 1.0: {{editedStory}}"))
  @Agent(outputKey = "score") def score(@V("editedStory") s: String): Double

trait CondDefaultFallback:
  @UserMessage(Array("Category '{{category}}' is unrecognized. Please use 'CREATIVE' or 'CRITICAL'."))
  @Agent(outputKey = "fallbackMessage") def handleFallback(@V("category") cat: String): String

trait TypedConditionalService:
  def run(
    @V("category") cat: String,
    @V("topic") topic: String,
    @V("editedStory") editedStory: String
  ): String

@main
def runConditionalExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val writer = AgenticServices.agentBuilder(classOf[CondWriter]).chatModel(model).build()
  val scorer = AgenticServices.agentBuilder(classOf[CondScorer]).chatModel(model).build()
  val fallbackAgent = AgenticServices.agentBuilder(classOf[CondDefaultFallback]).chatModel(model).build()

  val typedCond = AgenticServices.conditionalBuilder(classOf[TypedConditionalService])
    .subAgents(scope => scope.readState("category") == "CREATIVE", writer)
    .subAgents(scope => scope.readState("category") == "CRITICAL", scorer)
    .subAgents(
      scope =>
        val cat = scope.readState("category")
        cat != "CREATIVE" && cat != "CRITICAL"
      ,
      fallbackAgent
    )
    .output(scope =>
      if (scope.hasState("story")) scope.readState("story")
      else if (scope.hasState("score")) scope.readState("score").toString
      else scope.readState("fallbackMessage")
    )
    .build()

  println("Conditional Branching Workflow configuration built successfully.")

  try
    println("\nExecuting typed conditional workflow with CREATIVE category...")
    val creativeResult = typedCond.run("CREATIVE", "rainy Sunday afternoon", "")
    println(s"Creative Result: $creativeResult")

    println("\nExecuting typed conditional workflow with CRITICAL category...")
    val criticalResult = typedCond.run("CRITICAL", "", "It was a dark and stormy night.")
    println(s"Critical Result: $criticalResult")

    println("\nExecuting typed conditional workflow with UNRECOGNIZED category...")
    val unrecognizedResult = typedCond.run("INFORMATIONAL", "", "")
    println(s"Fallback Result: $unrecognizedResult")
  catch
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")
