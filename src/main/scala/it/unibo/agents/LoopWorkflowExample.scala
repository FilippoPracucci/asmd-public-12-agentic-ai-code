package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}

trait LoopScorer:
  @UserMessage(Array("Evaluate how detailed this story is. If it is detailed, return 0.9. If it is too short, return 0.2. Return ONLY a single double value like 0.2: {{editedStory}}"))
  @Agent(outputKey = "score") def score(@V("editedStory") s: String): Double

trait LoopEditor:
  @UserMessage(Array("Make this story twice as long and add more interesting adjectives: {{editedStory}}"))
  @Agent(outputKey = "editedStory") def edit(@V("editedStory") s: String): String

trait TypedLoopService:
  def run(@V("editedStory") story: String): String

@main
def runLoopExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val scorer = AgenticServices.agentBuilder(classOf[LoopScorer]).chatModel(model).build()
  val editor = AgenticServices.agentBuilder(classOf[LoopEditor]).chatModel(model).build()

  val typedLoop = AgenticServices.loopBuilder(classOf[TypedLoopService])
    .subAgents(scorer, editor)
    .maxIterations(3)
    .exitCondition(scope => scope.readState("score", 0.0) >= 0.8)
    .outputKey("editedStory")
    .build()

  println("Loop Workflow configuration built successfully.")

  try
    println("\nExecuting typed loop workflow...")
    val typedResult = typedLoop.run("A small cat sat on a comfortable mat.")
    println(s"Result: $typedResult")
  catch
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")
