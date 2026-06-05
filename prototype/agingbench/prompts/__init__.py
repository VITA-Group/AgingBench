"""Prompt registry for AgingBench.

Existing .txt files in this dir are compactor prompts (compact_lossy.txt etc.).
Python modules added here host scenario-aware system prompts and awareness
builders. All Python modules here are OPT-IN — the default agent prompt
remains ``core.agent.REACT_SYSTEM``.
"""
