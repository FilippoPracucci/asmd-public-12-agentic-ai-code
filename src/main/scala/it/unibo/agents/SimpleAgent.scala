package it.unibo.agents

import dev.langchain4j.agentic.{Agent, AgenticServices}
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{AiServices, UserMessage, V}

trait CreativeWriter:
  @UserMessage(Array("""
    You are a creative writer.
    Generate a story about the topic: {{topic}}.
    At most 50 words
  """))
  @Agent("Generates a story based on a given topic")
  def generateStory(@V("topic") topic: String): String

@main
def testCreativeWriter(): Unit =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(true)
    .logResponses(true)
    .build()
  val creativeWriter = AgenticServices
    .agentBuilder(classOf[CreativeWriter])
    .chatModel(model)
    .outputKey("story")
    .build()
