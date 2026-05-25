package it.unibo.verification

import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{AiServices, UserMessage}


val dataset = Seq(
  "What is a monad in programming languages?",
  "What does it mean for a type to be covariant?",
  "If A <: B, does List[A] <: List[B]? For example, if Cat <: Animal, is List[Cat] <: List[Animal]?",
  "How does monomorphization work?."
)

trait Evaluator:
  @UserMessage(Array("This is the {{question}} and this is the {{answer}}. Is the answer correct?"))
  def evaluate(question: String, answer: String): Boolean

trait QuestionAnsweringModel:
  @UserMessage(Array("Answer the following question: {{question}} - Short, to most 10 words."))
  def answer(question: String): String

def passK(evaluator: Evaluator,
          answerGenerator: QuestionAnsweringModel,
          question: String,
          k: Int,
          n: Int
         ): Double =
  require(n >= k, "n must be greater than or equal to k")

  val correctCount =
    Iterator.fill(n)(answerGenerator.answer(question))
      .count(answer => evaluator.evaluate(question, answer))

  def combinations(n: Int, k: Int): Double =
    if k < 0 || k > n then 0.0
    else if k == 0 || k == n then 1.0
    else
      val limit = math.min(k, n - k)
      (1 to limit).foldLeft(1.0) { (acc, i) =>
        acc * (n - i + 1) / i
      }

  val failedCount = n - correctCount

  // pass@k = 1 - C(n - c, k) / C(n, k)
  // where:
  // - n is the total number of samples
  // - c is the number of correct samples
  // - C(a, b) is "a choose b"
  if failedCount < k then 1.0 else 1.0 - (combinations(failedCount, k) / combinations(n, k))

@main
def passKForQuestions(key: String) =
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("gemma4:e2b")
    .logRequests(false)
    .logResponses(false)
    .build()

  val judgeModel = GoogleAiGeminiChatModel.builder()
    .apiKey(key)
    .modelName("gemini-3.5-flash")
    .temperature(0.0)
    .build()

  val evaluator: Evaluator = AiServices
    .builder(classOf[Evaluator])
    .chatModel(judgeModel)
    .build()

  val answerGenerator: QuestionAnsweringModel = AiServices
    .builder(classOf[QuestionAnsweringModel])
    .chatModel(model)
    .build()

  println("Question, Pass@1, Pass@5")
  dataset
    .map { question =>
      val passAt1 = passK(evaluator, answerGenerator, question, k = 1, n = 3)
      //val passAt5 = passK(evaluator, answerGenerator, question, k = 5, n = 10)
      f""""$question", $passAt1"""
    }.foreach(println)

