"""
agentic_framework.py
--------------------
Full Agentic AI Framework with Autonomous Decision-Making

This module provides:
- Tool-calling loop with OpenAI function calling
- Planning and reasoning capabilities
- Multi-agent orchestration
- Dynamic tool selection
- Self-reflection and error correction
- Agent memory and context management

The framework enables true agentic behavior where the AI:
1. Analyzes the task
2. Plans the approach
3. Selects appropriate tools
4. Executes actions autonomously
5. Reflects on results
6. Corrects errors and adapts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------

class ToolType(Enum):
    """Types of tools available to agents"""
    PRODUCT_SEARCH = "product_search"
    PRODUCT_DETAILS = "product_details"
    INVENTORY_CHECK = "inventory_check"
    ORDER_VALIDATION = "order_validation"
    RAG_SEARCH = "rag_search"
    PRICE_ANALYSIS = "price_analysis"
    CATEGORY_FILTER = "category_filter"
    CALCULATE = "calculate"
    DATA_AGGREGATE = "data_aggregate"


@dataclass
class Tool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    tool_type: ToolType
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        try:
            return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return {"error": str(e), "success": False}


# -----------------------------------------------------------------------------
# Agent State and Memory
# -----------------------------------------------------------------------------

@dataclass
class AgentThought:
    """Represents a single thought/reasoning step"""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None


@dataclass
class AgentMemory:
    """Maintains agent's working memory and history"""
    thoughts: List[AgentThought] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_thought(self, thought: AgentThought):
        """Add a reasoning step"""
        self.thoughts.append(thought)
    
    def add_tool_call(self, tool_name: str, args: Dict, result: Any):
        """Record a tool execution"""
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": args,
            "result": result
        })
    
    def get_summary(self) -> str:
        """Generate a summary of agent's process"""
        summary = []
        for thought in self.thoughts:
            summary.append(f"Step {thought.step}: {thought.thought}")
            if thought.action:
                summary.append(f"  Action: {thought.action}")
            if thought.observation:
                summary.append(f"  Result: {thought.observation}")
        return "\n".join(summary)


# -----------------------------------------------------------------------------
# Agentic Loop - ReAct Pattern (Reason + Act)
# -----------------------------------------------------------------------------

@dataclass
class AgenticExecutor:
    """
    Autonomous agent with planning, tool use, and reflection capabilities.
    
    Uses the ReAct (Reasoning + Acting) pattern:
    1. Thought: Reason about what to do next
    2. Action: Select and execute a tool
    3. Observation: Observe the result
    4. Reflection: Evaluate if goal is achieved or continue
    """
    
    client: OpenAI
    model: str = "gpt-4o"
    tools: List[Tool] = field(default_factory=list)
    memory: AgentMemory = field(default_factory=AgentMemory)
    max_iterations: int = 10
    temperature: float = 0.1
    
    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use"""
        self.tools.append(tool)
        logger.info(f"Registered tool: {tool.name}")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions"""
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        return f"""You are an autonomous AI agent with access to multiple tools. Your goal is to help users by planning and executing tasks step-by-step.

AVAILABLE TOOLS:
{tools_desc}

REASONING FRAMEWORK (ReAct Pattern):
For each step, you must:
1. THINK: Analyze the current situation and decide what to do next
2. ACT: Select the most appropriate tool and specify parameters
3. OBSERVE: Review the tool's output
4. REFLECT: Determine if the goal is achieved or if more steps are needed

IMPORTANT RULES:
- Always explain your reasoning before taking action
- Use tools systematically - don't guess information you can retrieve
- If a tool fails, try alternative approaches
- When you have sufficient information to answer, provide a complete response
- Be efficient - don't make unnecessary tool calls
- Validate data before using it in subsequent steps

OUTPUT FORMAT:
Structure your response as:
```
Thought: [Your reasoning about what to do]
Action: [Tool name to use]
Action Input: [Parameters as JSON]
```

After receiving observations, continue with:
```
Observation: [What you learned]
Thought: [Next reasoning step]
```

When task is complete:
```
Final Answer: [Complete response to user]
```"""
    
    def _parse_agent_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        Parse agent's response to extract thought, action, and action input.
        
        Returns:
            (thought, action, action_input)
        """
        thought = None
        action = None
        action_input = None
        
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                # Try to parse JSON
                try:
                    # Remove markdown code blocks if present
                    input_str = input_str.replace("```json", "").replace("```", "").strip()
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse action input: {input_str}")
                    action_input = {}
        
        return thought, action, action_input
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                result = tool.execute(**tool_input)
                self.memory.add_tool_call(tool_name, tool_input, result)
                return result
        
        return {"error": f"Tool {tool_name} not found", "success": False}
    
    def _should_stop(self, response: str) -> bool:
        """Check if agent has reached final answer"""
        return "Final Answer:" in response or "FINAL ANSWER:" in response.upper()
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response"""
        for line in response.split("\n"):
            if "Final Answer:" in line or "FINAL ANSWER:" in line.upper():
                return line.split(":", 1)[1].strip()
        return response
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task autonomously using available tools.
        
        Args:
            task: The task description
            context: Optional context information
            
        Returns:
            Dictionary with result, memory, and execution trace
        """
        if context:
            self.memory.context = context
        
        # Build conversation history
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            messages.append({
                "role": "system",
                "content": f"Additional Context:\n{context_str}"
            })
        
        iteration = 0
        final_answer = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
            
            # Get agent's response
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000
                )
                agent_response = response.choices[0].message.content
                
                # Add to conversation
                messages.append({"role": "assistant", "content": agent_response})
                
                logger.debug(f"Agent response: {agent_response}")
                
                # Check if done
                if self._should_stop(agent_response):
                    final_answer = self._extract_final_answer(agent_response)
                    logger.info("Agent reached final answer")
                    break
                
                # Parse response
                thought, action, action_input = self._parse_agent_response(agent_response)
                
                if not action:
                    # If no action specified, ask agent to continue
                    messages.append({
                        "role": "user",
                        "content": "Please specify an action to take or provide a Final Answer."
                    })
                    continue
                
                # Record thought
                agent_thought = AgentThought(
                    step=iteration,
                    thought=thought or "No explicit thought",
                    action=action,
                    action_input=action_input
                )
                
                # Execute tool
                logger.info(f"Executing tool: {action} with input: {action_input}")
                observation = self._execute_tool(action, action_input or {})
                observation_str = json.dumps(observation, indent=2) if observation else "No result"
                
                agent_thought.observation = observation_str
                self.memory.add_thought(agent_thought)
                
                # Provide observation to agent
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation_str}\n\nContinue with your next thought and action, or provide the Final Answer if you have enough information."
                })
                
            except Exception as e:
                logger.error(f"Error in agent iteration {iteration}: {e}")
                final_answer = f"Agent execution failed: {str(e)}"
                break
        
        if final_answer is None:
            final_answer = "Agent reached maximum iterations without completing the task."
        
        return {
            "success": final_answer is not None,
            "result": final_answer,
            "iterations": iteration,
            "memory": self.memory,
            "execution_trace": self.memory.get_summary(),
            "tool_calls": self.memory.tool_calls
        }


# -----------------------------------------------------------------------------
# Planner Agent - High-Level Task Planning
# -----------------------------------------------------------------------------

@dataclass
class TaskPlan:
    """Represents a high-level task plan"""
    goal: str
    steps: List[str]
    required_tools: List[str]
    estimated_complexity: str  # "simple", "medium", "complex"


class PlannerAgent:
    """
    High-level planner that breaks down complex tasks into steps.
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def create_plan(self, task: str, available_tools: List[str]) -> TaskPlan:
        """
        Create a high-level plan for accomplishing a task.
        
        Args:
            task: The task to plan
            available_tools: List of available tool names
            
        Returns:
            TaskPlan object
        """
        tools_list = "\n".join([f"- {tool}" for tool in available_tools])
        
        prompt = f"""You are a planning agent. Given a task and available tools, create a step-by-step plan.

TASK: {task}

AVAILABLE TOOLS:
{tools_list}

Create a structured plan with:
1. Clear sequential steps
2. Which tools are needed
3. Estimated complexity (simple/medium/complex)

Respond in JSON format:
{{
  "steps": ["step 1", "step 2", ...],
  "required_tools": ["tool1", "tool2", ...],
  "complexity": "simple|medium|complex"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert planning agent. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                
                return TaskPlan(
                    goal=task,
                    steps=plan_data.get("steps", []),
                    required_tools=plan_data.get("required_tools", []),
                    estimated_complexity=plan_data.get("complexity", "medium")
                )
        except Exception as e:
            logger.error(f"Planning failed: {e}")
        
        # Fallback plan
        return TaskPlan(
            goal=task,
            steps=["Analyze the task", "Execute required actions", "Provide result"],
            required_tools=[],
            estimated_complexity="medium"
        )


# -----------------------------------------------------------------------------
# Multi-Agent Orchestrator
# -----------------------------------------------------------------------------

class AgentRole(Enum):
    """Specialized agent roles"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


@dataclass
class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents working together.
    
    Workflow:
    1. Planner: Creates high-level plan
    2. Executor: Implements the plan using tools
    3. Validator: Checks results for correctness
    4. Synthesizer: Combines results into final answer
    """
    
    client: OpenAI
    model: str = "gpt-4o"
    tools: List[Tool] = field(default_factory=list)
    
    def __post_init__(self):
        self.planner = PlannerAgent(self.client, self.model)
        self.executor = AgenticExecutor(self.client, self.model, self.tools)
    
    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools.append(tool)
        self.executor.register_tool(tool)
    
    def execute_with_planning(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a task with full planning and orchestration.
        
        Args:
            task: The task to execute
            context: Optional context
            
        Returns:
            Comprehensive result with plan, execution trace, and answer
        """
        logger.info(f"Multi-agent orchestration started for task: {task}")
        
        # Step 1: Planning
        available_tools = [tool.name for tool in self.tools]
        plan = self.planner.create_plan(task, available_tools)
        logger.info(f"Plan created with {len(plan.steps)} steps, complexity: {plan.estimated_complexity}")
        
        # Step 2: Execution
        execution_result = self.executor.execute(task, context)
        
        # Step 3: Validation (check if result makes sense)
        is_valid = self._validate_result(task, execution_result["result"])
        
        # Step 4: Synthesis
        final_result = {
            "success": execution_result["success"] and is_valid,
            "answer": execution_result["result"],
            "plan": {
                "steps": plan.steps,
                "complexity": plan.estimated_complexity,
                "required_tools": plan.required_tools
            },
            "execution": {
                "iterations": execution_result["iterations"],
                "tool_calls": len(execution_result["tool_calls"]),
                "trace": execution_result["execution_trace"]
            },
            "validation": {
                "passed": is_valid,
                "confidence": "high" if is_valid else "low"
            }
        }
        
        logger.info(f"Multi-agent orchestration completed. Valid: {is_valid}")
        return final_result
    
    def _validate_result(self, task: str, result: str) -> bool:
        """
        Validate if the result adequately answers the task.
        
        Returns:
            True if result is valid
        """
        if not result or "error" in result.lower():
            return False
        
        # Basic validation - result should be substantial
        if len(result.strip()) < 10:
            return False
        
        # Could add more sophisticated validation with another LLM call
        return True
