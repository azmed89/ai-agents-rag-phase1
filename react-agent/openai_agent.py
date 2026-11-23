"""Structured tool-calling demo using the OpenAI Agents SDK."""
import asyncio
import getpass
import os
import shlex
import sys
from pathlib import Path
from agents import Agent, ModelSettings, Runner, function_tool

BASE = Path(__file__).resolve().parent
DEMO = (BASE / 'demo').resolve()
TASKS = [
    'List the files in ./demo and summarise what this mini-project does',
    'Read README.md in ./demo and tell me which file handles orders',
    "Check if there is a TODO file — if not, say so, don't invent one",
]

def safe_path(path: str) -> Path:
    result = (BASE / path).resolve()
    if not result.is_relative_to(DEMO):
        raise ValueError('Only paths inside demo/ are allowed')
    return result

@function_tool
def read_file(path: str) -> str:
    """Read a UTF-8 file inside demo/."""
    try:
        return safe_path(path).read_text(encoding='utf-8')[:4000]
    except (OSError, ValueError) as exc:
        return f'ERROR: {exc}'

@function_tool
def list_dir(path: str = 'demo') -> str:
    """List a directory inside demo/."""
    try:
        return '\n'.join(sorted(p.name + ('/' if p.is_dir() else '') for p in safe_path(path).iterdir())) or '(empty)'
    except (OSError, ValueError) as exc:
        return f'ERROR: {exc}'

@function_tool
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 file inside demo/."""
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f'Wrote {len(content)} characters to {path}'
    except (OSError, ValueError) as exc:
        return f'ERROR: {exc}'

@function_tool
def run_command(command: str) -> str:
    """List a directory with 'ls demo' or 'dir demo'. No shell is executed."""
    try:
        args = shlex.split(command)
        if len(args) != 2 or args[0] not in ('ls', 'dir'):
            return 'ERROR: only ls or dir with one demo/ path is allowed'
        return '\n'.join(sorted(p.name + ('/' if p.is_dir() else '') for p in safe_path(args[1]).iterdir())) or '(empty)'
    except (OSError, ValueError) as exc:
        return f'ERROR: {exc}'

async def main() -> None:
    if not os.environ.get('OPENAI_API_KEY'):
        try:
            key = getpass.getpass('Enter your OpenAI API key (input is hidden): ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nNo API key entered.', file=sys.stderr)
            return
        if not key:
            print('No API key entered.', file=sys.stderr)
            return
        os.environ['OPENAI_API_KEY'] = key
    agent = Agent(name='Demo assistant', model=os.environ.get('OPENAI_MODEL', 'gpt-4.1'),
        instructions='Inspect demo/ with tools before answering. Report missing files honestly; do not invent contents. Do not modify files unless requested.',
        model_settings=ModelSettings(temperature=0), tools=[read_file, list_dir, run_command, write_file])
    tasks = [' '.join(sys.argv[1:])] if len(sys.argv) > 1 else TASKS
    for task in tasks:
        for run in range(1, 4):
            print(f'\nTask: {task}\nRun {run}/3', flush=True)
            result = await Runner.run(agent, task, max_turns=10)
            print(result.final_output, flush=True)

if __name__ == '__main__':
    asyncio.run(main())
