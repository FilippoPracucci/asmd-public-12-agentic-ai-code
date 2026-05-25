package it.unibo.tools

import dev.langchain4j.agent.tool.{P, Tool}

class MathModule:
  @Tool(name="sum", value = Array("Calcalate the sum of two numbers"))
  def sum(@P("first number") left: Double, @P("second number") right: Double): Double =
    println(s"[MathModule Tool] sum($left, $right) called")
    left + right

  @Tool(name="subtract", value = Array("Calcalate the difference of two numbers"))
  def subtract(@P("first number") left: Double, @P("second number") right: Double): Double =
    println(s"[MathModule Tool] subtract($left, $right) called")
    left - right

  @Tool(name="multiply", value = Array("Calcalate the product of two numbers"))
  def multiply(@P("first number") left: Double, @P("second number") right: Double): Double =
    println(s"[MathModule Tool] multiply($left, $right) called")
    left * right

  @Tool(name="divide", value = Array("Calcalate the quotient of two numbers"))
  def divide(@P("first number") left: Double, @P("second number") right: Double): Double =
    println(s"[MathModule Tool] divide($left, $right) called")
    left / right

