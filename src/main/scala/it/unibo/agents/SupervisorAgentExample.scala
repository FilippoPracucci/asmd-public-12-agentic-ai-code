package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.agentic.supervisor.SupervisorResponseStrategy
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{UserMessage, V}
import java.util.HashMap

// 1. Math Agent
trait SupervisorMathAgent:
  @UserMessage(Array("""
    You are a math expert. Solve this math problem: {{problem}}.
    Return the numeric answer clearly.
  """))
  @Agent(outputKey = "mathResult", description = "Solves arithmetic calculations and basic mathematical problems")
  def solveMath(@V("problem") problem: String): String

// 2. Translation Agent
trait SupervisorTranslationAgent:
  @UserMessage(Array("""
    You are a professional translator. Translate this text: "{{text}}"
    Translate it into the target language: {{language}}.
    Return only the translated text and nothing else.
  """))
  @Agent(outputKey = "translationResult", description = "Translates any given English text into another target language (e.g. Italian, Spanish, French)")
  def translate(@V("text") text: String, @V("language") language: String): String

// 3. Summarizer Agent
trait SupervisorSummarizerAgent:
  @UserMessage(Array("""
    You are a professional editor. Summarize the following text in exactly one concise sentence:
    "{{textToSummarize}}"
    Return only the single sentence summary.
  """))
  @Agent(outputKey = "summaryResult", description = "Summarizes long text or reports into a single short sentence")
  def summarize(@V("textToSummarize") textToSummarize: String): String

@main
def runSupervisorExample(): Unit =
  println("--- Initializing local Ollama model (gemma4:e2b) for supervisor planner and agents ---")
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  // Build individual agents
  val mathAgent = AgenticServices.agentBuilder(classOf[SupervisorMathAgent])
    .chatModel(model)
    .build()

  val translationAgent = AgenticServices.agentBuilder(classOf[SupervisorTranslationAgent])
    .chatModel(model)
    .build()

  val summarizerAgent = AgenticServices.agentBuilder(classOf[SupervisorSummarizerAgent])
    .chatModel(model)
    .build()

  println("\n==============================================")
  println("PART 2: INITIALIZING SUPERVISOR AGENT")
  println("==============================================")

  // Configure a Supervisor Agent using the supervisorBuilder.
  // The supervisor acts as an LLM-driven planner orchestrating the subagents.
  val supervisor = AgenticServices.supervisorBuilder()
    .chatModel(model)
    .subAgents(mathAgent, translationAgent, summarizerAgent)
    .supervisorContext("""
      Always prefer delegating to the appropriate specialized subagent.
      IMPORTANT:
      If a subagent has already been called and returned a result, DO NOT call it again.
      Immediately invoke 'done' with the final answer.
    """)
    .responseStrategy(SupervisorResponseStrategy.SUMMARY)
    .maxAgentsInvocations(2)
    .build()

  // Task 1: Ask the supervisor a math question
  println("\n--- Task 1: Mathematical Problem ---")
  val query1 = "Calculate 125 multiplied by 8 and add 50."
  println(s"User query: '$query1'")
  val result1: String = supervisor.invoke(query1)
  println("[Supervisor Summary Result]:")
  println(result1)

  // Task 2: Ask the supervisor a translation question
  println("\n--- Task 2: Language Translation ---")
  val query2 = "Translate 'Programming in Scala 3 with agentic frameworks is incredibly satisfying!' into Italian."
  println(s"User query: '$query2'")
  val result2: String = supervisor.invoke(query2)
  println("[Supervisor Summary Result]:")
  println(result2)

  // Task 3: Ask the supervisor a text summarization
  println("\n--- Task 3: Text Summarization ---")
  val longText = "Agentic AI refers to systems where large language models function as a dynamic reasoning core capable of selecting actions, " +
    "utilizing external tools, maintaining conversational state via memory, and evaluating outcomes. " +
    "Rather than following rigid pipelines, pure agents decide the sequence of steps dynamically based on observations, " +
    "enabling adaptive problem-solving."
  val query3 = s"Summarize this text into one sentence: $longText"
  println(s"User query: '$query3'")
  val result3: String = supervisor.invoke(query3)
  println("[Supervisor Summary Result]:")
  println(result3)
