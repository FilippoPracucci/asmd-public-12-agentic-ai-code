# Agentic AI - Architectures, Patterns & Evaluation
## Practical Guide to Core Agentic Patterns and LLM Workflows in Scala 3

This repository provides a progressive guide to implementing Agentic AI patterns using Scala 3, langchain4j, and langchain4j-agentic. 
The project covers structured output parsing, tool integration, conversational memory,
retrieval-augmented generation (RAG), 
and multi-agent system orchestration.

---

## Requirements

The codebase leverages local Small Language Models (SLMs) and Embeddings via Ollama, 
along with Google Gemini for evaluation.

1. Install Ollama by following the instructions in the [official repository](https://ollama.com/download).
2. Download the required models:
   ```bash
   # Small Language Model for reasoning and text generation
   ollama pull gemma4:e2b

   # Models optimized for tool execution and function calling
   ollama pull qwen3.5:4b
   ollama pull qwen2.5:latest

   # Embedding model for RAG and vector similarity search
   ollama pull ibm/granite-embedding:30m
   ```
3. Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/api-keys) to run the verification and evaluation suite.

---

## Project Structure

The project code is organized progressively under the `it.unibo` package:

### 1. AI Services and Structured I/O (`it.unibo.services`)
This package contains examples of basic classification and structured output parsing. `SentimentAnalyzer.scala` implements a binary classifier returning a standard Scala `Boolean` for text sentiment. `TicTacToe.scala` shows a structured LLM-in-the-loop setup where model moves are parsed into typed `Move` instances using native JSON response formatting.

### 2. Tools and Function Calling (`it.unibo.tools`)
This package demonstrates how to extend models with external computations. `MathModule.scala` defines double-precision math operations annotated with `@Tool`. `ToolExample.scala` shows how tool schemas are programmatically generated, and `ToolsViaZeroShot.scala` integrates these math tools with an active AI service to solve composite queries.

### 3. Memory and Retrieval-Augmented Generation (`it.unibo.memory`)
This package covers conversational state and knowledge retrieval. `MemoryExample.scala` implements state retention using `MessageWindowChatMemory` across multiple turns. `RagExample.scala` demonstrates document ingestion, local vector storage, and retrieving relevant facts from local resources.

### 4. Advanced Agentic Architectures (`it.unibo.agents`)
This package implements multi-agent coordination and workflows. `ScopeExample.scala` uses `AgenticScope` as a shared blackboard for state tracking and failure recovery. `UnifiedAgentExample.scala` coordinates research, math, and formatting subagents under a central supervisor. The package also includes micro-examples for common execution flows: `SequentialWorkflowExample.scala` for pipelines, `LoopWorkflowExample.scala` for iterative refinement, `ParallelWorkflowExample.scala` and `ParallelMapperExample.scala` for concurrent executions, `ConditionalWorkflowExample.scala` for dynamic branching, and `SupervisorWorkflowExample.scala` for central routing.

### 5. Evaluation and Verification (`it.unibo.verification`)
This package provides evaluation metrics for agent outputs. `Evaluator.scala` uses an LLM-as-a-Judge technique to evaluate generated answers and compute the unbiased `pass@k` metric.

### 6. Practice Exercise (`it.unibo.exercise`)
This package provides the navigation exercise. See the README.MD file inside the `exercise` folder for detailed instructions on how to implement the tools and agent configuration to successfully navigate the robot to the target position while avoiding obstacles.
