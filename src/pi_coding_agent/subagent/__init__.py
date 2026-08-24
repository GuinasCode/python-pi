"""Subagent orchestration for Pi coding agent.

Three modes:
  single    { agent, task }               one subagent, one task
  parallel  { tasks: [{agent, task},...] } up to 8 agents, 4 concurrent
  chain     { chain: [{agent, task},...] } sequential, {previous} placeholder
"""

from __future__ import annotations

from pi_coding_agent.subagent.agent_def import AgentDef, discover_agents
from pi_coding_agent.subagent.registry import SubagentHandle, SubagentRegistry, SubagentResult
from pi_coding_agent.subagent.tool import create_subagent_tool

__all__ = [
    "AgentDef",
    "SubagentHandle",
    "SubagentRegistry",
    "SubagentResult",
    "create_subagent_tool",
    "discover_agents",
]
