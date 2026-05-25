package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.agentic.scope.{AgenticScope, DefaultAgenticScope, AgenticScopeSerializer, ResultWithAgenticScope}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}
import java.util.HashMap

// 1. Defining specialized traits for agents.
// The output of each agent is automatically written to the scope using the key specified in outputKey.
trait ScopeCreativeWriter:
  @UserMessage(Array("""
    You are a creative writer.
    Generate a story about the topic: {{topic}}.
    Make it extremely short (at most 2 sentences).
  """))
  @Agent(outputKey = "story", description = "Generates a short story based on a given topic")
  def generateStory(@V("topic") topic: String): String

trait ScopeAudienceEditor:
  @UserMessage(Array("""
    You are a professional editor.
    Analyze and rewrite the following story to better align with the target audience: {{audience}}.
    The story: "{{story}}"
    Keep it at most 2 sentences.
  """))
  @Agent(outputKey = "editedStory", description = "Edits a story to better fit a given audience")
  def editStory(@V("story") story: String, @V("audience") audience: String): String

@main
def runScopeExample(): Unit =
  println("==============================================")
  println("INITIALIZING AGENTS AND WORKFLOW WITH AGENTIC SCOPE")
  println("==============================================")

  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val creativeWriter = AgenticServices.agentBuilder(classOf[ScopeCreativeWriter])
    .chatModel(model)
    .build()

  val audienceEditor = AgenticServices.agentBuilder(classOf[ScopeAudienceEditor])
    .chatModel(model)
    .build()

  // Chaining them into a sequential workflow
  val workflow = AgenticServices.sequenceBuilder()
    .subAgents(creativeWriter, audienceEditor)
    .outputKey("editedStory")
    .build()

  // Prepare input state
  val inputs = new HashMap[String, Any]()
  inputs.put("topic", "Artificial Intelligence in daily life")
  inputs.put("audience", "children")

  println("Running sequential workflow...")
  // We use invokeWithAgenticScope instead of invoke to get the resulting AgenticScope
  val resultWithScope = workflow.invokeWithAgenticScope(inputs)
  val finalResult = resultWithScope.result()
  val scope: AgenticScope = resultWithScope.agenticScope()

  println(s"\nWorkflow Output: $finalResult")

  println("\n==============================================")
  println("1. SHARED BLACKBOARD FEATURE")
  println("==============================================")
  println("Reading state directly from the shared blackboard:")
  println(s" - topic:       ${scope.readState("topic")}")
  println(s" - audience:    ${scope.readState("audience")}")
  println(s" - story:       ${scope.readState("story")}")
  println(s" - editedStory: ${scope.readState("editedStory")}")

  println("\n==============================================")
  println("2. AUTOMATIC REGISTRY FEATURE")
  println("==============================================")
  println("The scope automatically logs the sequence of agent executions:")
  val invocations = scope.agentInvocations()
  invocations.forEach { inv =>
    println(s" - Agent: ${inv.agentName()}")
    println(s"   Inputs:  ${inv.input()}")
    println(s"   Outputs: ${inv.output()}")
  }

  println("\n==============================================")
  println("3. PERSISTENCE & RECOVERY FEATURE")
  println("==============================================")
  // Casting to DefaultAgenticScope is required to use AgenticScopeSerializer
  val defaultScope = scope.asInstanceOf[DefaultAgenticScope]
  
  // Serialize the state to JSON
  println("Serializing AgenticScope to JSON...")
  val jsonRepresentation = AgenticScopeSerializer.toJson(defaultScope)
  println(s"Serialized JSON length: ${jsonRepresentation.length} characters")
  
  // Deserialize back to a new AgenticScope instance
  println("Deserializing AgenticScope back from JSON...")
  val restoredScope: DefaultAgenticScope = AgenticScopeSerializer.fromJson(jsonRepresentation)
  
  println("\nVerifying restored state:")
  println(s" - Restored memoryId:  ${restoredScope.memoryId()}")
  println(s" - Restored topic:     ${restoredScope.readState("topic")}")
  println(s" - Restored story:     ${restoredScope.readState("story")}")
  println(s" - Restored invocations count: ${restoredScope.agentInvocations().size()}")
