package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}
import java.util.HashMap
import scala.jdk.CollectionConverters._

// 1. Creative Writer Agent (Renamed to WorkflowCreativeWriter)
trait WorkflowCreativeWriter:
  @UserMessage(Array("""
    You are a creative writer.
    Generate a story about the topic: {{topic}}.
    Make it extremely short (at most 2 sentences).
  """))
  @Agent(outputKey = "story", description = "Generates a short story based on a given topic")
  def generateStory(@V("topic") topic: String): String

// 2. Audience Editor Agent (Renamed to WorkflowAudienceEditor)
trait WorkflowAudienceEditor:
  @UserMessage(Array("""
    You are a professional editor.
    Analyze and rewrite the following story to better align with the target audience: {{audience}}.
    The story: "{{story}}"
    Keep it at most 2 sentences.
  """))
  @Agent(outputKey = "editedStory", description = "Edits a story to better fit a given audience")
  def editStory(@V("story") story: String, @V("audience") audience: String): String

// 3. Style Scorer Agent (Renamed to WorkflowStyleScorer)
trait WorkflowStyleScorer:
  @UserMessage(Array("""
    You are a style reviewer.
    Score how well the following story aligns with the {{style}} style.
    Return ONLY a floating point number between 0.0 (poor alignment) and 1.0 (perfect alignment).
    Story: "{{editedStory}}"
  """))
  @Agent(outputKey = "score", description = "Evaluates the style alignment and returns a Double score")
  def scoreStyle(@V("editedStory") editedStory: String, @V("style") style: String): Double

// 4. Style Editor Agent (Renamed to WorkflowStyleEditor)
trait WorkflowStyleEditor:
  @UserMessage(Array("""
    You are a professional style editor.
    Revise the following story to make it align perfectly with the {{style}} style.
    The story: "{{editedStory}}"
    Keep it at most 2 sentences.
  """))
  @Agent(outputKey = "editedStory", description = "Edits the story to better fit the requested style")
  def applyStyle(@V("editedStory") editedStory: String, @V("style") style: String): String

@main
def runWorkflowExample(): Unit =
  println("--- Initializing local Ollama model (gemma4:e2b) ---")
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(true)
    .logResponses(true)
    .build()

  // Build the individual agents
  val creativeWriter = AgenticServices.agentBuilder(classOf[WorkflowCreativeWriter])
    .chatModel(model)
    .build()

  val audienceEditor = AgenticServices.agentBuilder(classOf[WorkflowAudienceEditor])
    .chatModel(model)
    .build()

  val styleScorer = AgenticServices.agentBuilder(classOf[WorkflowStyleScorer])
    .chatModel(model)
    .build()

  val styleEditor = AgenticServices.agentBuilder(classOf[WorkflowStyleEditor])
    .chatModel(model)
    .build()

  println("\n==============================================")
  println("PART 1: RUNNING DETERMINISTIC SEQUENTIAL WORKFLOW")
  println("==============================================")

  // Combine CreativeWriter and AudienceEditor into a sequential workflow
  val sequentialWorkflow = AgenticServices.sequenceBuilder()
    .subAgents(creativeWriter, audienceEditor)
    .outputKey("editedStory")
    .build()
  
  // Prepare initial inputs in a HashMap for Java integration
  val initialInputs = new HashMap[String, Any]()
  initialInputs.put("topic", "Agentic AI in Scala")
  initialInputs.put("audience", "excited software engineering students")

  // Execute the sequential workflow
  println(s"Sending topic: 'Agentic AI in Scala' for audience: 'excited software engineering students'...")
  val resultObj: Object = sequentialWorkflow.invoke(initialInputs)
  val editedStory = resultObj.asInstanceOf[String]

  println("\n[Audience-Adapted Story]:")
  println(editedStory)

  println("\n==============================================")
  println("PART 2: RUNNING ITERATIVE LOOP WORKFLOW")
  println("==============================================")

  // Combine StyleScorer and StyleEditor into a loop workflow that continues
  // until the style score meets or exceeds 0.8 (or max 4 iterations)
  val loopWorkflow = AgenticServices.loopBuilder()
    .subAgents(styleScorer, styleEditor)
    .maxIterations(4)
    .exitCondition(scope => scope.readState("score", 0.0) >= 0.8)
    .outputKey("editedStory")
    .build()

  // Put the current story and target style in the state
  val loopInputs = new HashMap[String, Any]()
  loopInputs.put("editedStory", editedStory)
  loopInputs.put("style", "extremely poetic and dramatic")

  println("Starting style improvement loop for style: 'extremely poetic and dramatic'...")
  val loopResultObj: Object = loopWorkflow.invoke(loopInputs)
  val finalStory = loopResultObj.asInstanceOf[String]

  println("\n[Final Polished Story after Loop]:")
  println(finalStory)
