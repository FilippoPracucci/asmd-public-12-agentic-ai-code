package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}

trait SeqWriter:
  @UserMessage(Array("Write an extremely short story (1 sentence) about: {{topic}}"))
  @Agent(outputKey = "story") def write(@V("topic") t: String): String

trait SeqEditor:
  @UserMessage(Array("Rewrite this story to make it sound like a space-opera legend: {{story}}"))
  @Agent(outputKey = "editedStory") def edit(@V("story") s: String): String

trait TypedSeqService:
  def run(@V("topic") topic: String): String

@main
def runSequentialExample(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val writer = AgenticServices.agentBuilder(classOf[SeqWriter]).chatModel(model).build()
  val editor = AgenticServices.agentBuilder(classOf[SeqEditor]).chatModel(model).build()

  val typedSeq = AgenticServices.sequenceBuilder(classOf[TypedSeqService])
    .subAgents(writer, editor)
    .outputKey("editedStory")
    .build()

  println("Sequential Workflow configuration built successfully.")

  try
    println("\nExecuting typed sequential workflow...")
    val typedResult = typedSeq.run("a tiny brave robot on Mars")
    println(s"Typed Result: $typedResult")
  catch
    case e: Exception =>
      println(s"\nCould not run workflow execution. Ensure Ollama is running with gemma4:e2b locally.\nError: ${e.getMessage}")

