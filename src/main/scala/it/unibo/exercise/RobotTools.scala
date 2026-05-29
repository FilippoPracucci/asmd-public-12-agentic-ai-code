package it.unibo.exercise

import dev.langchain4j.agent.tool.Tool
import it.unibo.exercise.AgentsAction.*

/**
 * EXERCISE: Implement the tool integration for the Robot.
 * These methods should be annotated with @Tool from LangChain4j, allowing the LLM-based agent
 * to interact with and inspect the Environment.
 * 
 * Your tasks:
 * 1. Understand the main actions the robot can perform (move up/down/left/right, hold, release) and how to retrieve the environment status.
 * 2. Implement each tool's logic by invoking the corresponding actions on the `env` (Environment) object.
 * 3. Return the resulting String/Call statue from each tool call so the agent is informed of the result.
 */
class RobotTools(val env: Environment):
  @Tool(name = "moveUp", value = Array("Move the robot up in the grid, meaning in the adjacent cell above."))
  def moveUp(): String = env.step(MoveUp)

  @Tool(name = "moveDown", value = Array("Move the robot down in the grid, meaning in the adjacent cell below."))
  def moveDown(): String = env.step(MoveDown)

  @Tool(name = "moveLeft", value = Array("Move the robot left in the grid, meaning in the adjacent left cell."))
  def moveLeft(): String = env.step(MoveLeft)

  @Tool(name = "moveRight", value = Array("Move the robot right in the grid, meaning in the adjacent right cell."))
  def moveRight(): String = env.step(MoveRight)

  @Tool(name = "hold", value = Array("The robot grab the object and hold it."))
  def hold(): String = env.step(Hold)

  @Tool(name = "release", value = Array("The robot release the object that it was holding."))
  def release(): String = env.step(Release)
