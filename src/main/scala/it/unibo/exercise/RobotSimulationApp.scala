package it.unibo.exercise

import dev.langchain4j.memory.chat.MessageWindowChatMemory
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.service.{AiServices, SystemMessage, UserMessage, V}

/**
 * EXERCISE: Define the RobotAgent trait.
 * 
 * Your tasks:
 * 1. Define the system instructions for the robot agent using @SystemMessage.
 *    The instruction should guide the agent to act as a navigation AI that uses tools
 *    to help the robot reach a goal position.
 * 2. Define the user message format using @UserMessage.
 * 3. Define the next method with a RobotState argument annotated with @V.
 */
trait RobotAgent:
  @SystemMessage(Array("You are a robot agent that can perform actions in a grid environment to reach the goal."))
  def foo: Unit = {}

  
  // For compilation purposes while starting the exercise, we leave a placeholder signature.
  // You can modify this as needed when implementing.
  @UserMessage(Array("""
      Given the current robot state, what is your next move? Robot state: {{state}}. Consider the constraints:
        - position._1 >= 0 and <= environment.width
        - position._2 >= 0 and <= environment.height
        - holdingObject true to release it or false to grab it
  """))
  def next(@V("state") state: RobotState): AgentsAction = ???

@main
def runRobotSimulation(): Unit =
  println("==========================================================")
  println("  ROBOT SIMULATION WITH AGENTIC AI (EXERCISE)             ")
  println("==========================================================")

  // 1. Set up the starting robot and environment
  val initialRobotState = RobotState(
    position = (0.0, 0.0),
    orientation = 0.0,
    holdingObject = false
  )
  val robot = Robot(id = "robo-01", state = initialRobotState, name = "Explorer-1")

  // Create a 4x4 grid.
  // Start at (0,0), Goal at (3,3). Obstacles block some paths.
  val env = Environment(
    width = 4,
    height = 4,
    goal = (3.0, 3.0),
    obstacles = Set((1.0, 1.0), (1.0, 2.0), (2.0, 1.0)),
    robot = robot
  )

  // TODO: 2. Build the Ollama Chat Model.
  // We recommend 'qwen3.5:4b' or 'gemma4:e2b' for this exercise, but you can experiment with others.
  // Ensure your Ollama server is running locally (http://localhost:11434).
  /*
  println("Initializing Ollama Chat Model (qwen3.5:4b)...")
  val model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("qwen3.5:4b")
    .temperature(0.0) // Low temperature for deterministic planning
    .logRequests(false)
    .logResponses(false)
    .build()
  */

  // TODO: 3. Bind the environment to your tools helper

  // TODO: 4. Build the AI Service with tools and memory window

  // 5. Start the external step-by-step simulation loop
  println("\nStarting navigation loop...")
  var stepCount = 0
  val maxSteps = 20

  while (!env.isGoalReached && stepCount < maxSteps) {
    stepCount += 1
    println(s"\n--- STEP $stepCount ---")
    println(env.printGrid())
    println(s"Current Robot State: ${env.robot.state}")
    
    // TODO: 6. Invoke the agent passing the current state.
    
    // Brief sleep to visualize the progress smoothly
    Thread.sleep(1000)
  }

  println("\n==========================================================")
  if (env.isGoalReached)
    println("🎉 GOAL REACHED! The robot arrived safely at the goal position (3.0, 3.0)!")
  else
    println("❌ SIMULATION STOPPED / TIMEOUT.")
  println("==========================================================")
  println("Final Environment State:")
  println(env.printGrid())
  println("Exercise setup complete! Ready for your implementation.")

