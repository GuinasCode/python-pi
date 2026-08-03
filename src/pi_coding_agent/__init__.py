"""CLI argument parsing and entry point for the Pi coding agent.

Mirrors packages/coding-agent/src/cli/args.ts and packages/coding-agent/src/main.ts.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

APP_NAME = "pi"
CONFIG_DIR_NAME = ".pi"

VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
VALID_MODES = ("text", "json", "rpc")
VALID_UI_MODES = ("regular", "fullscreen")


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


@dataclass
class Args:
    """Parsed CLI arguments."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: list[str] | None = None
    thinking: str | None = None
    continue_session: bool = False
    resume: bool = False
    help: bool = False
    version: bool = False
    mode: str | None = None
    name: str | None = None
    no_session: bool = False
    session: str | None = None
    session_id: str | None = None
    fork: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: bool = False
    no_builtin_tools: bool = False
    extensions: list[str] | None = None
    no_extensions: bool = False
    print: bool = False
    export: str | None = None
    no_skills: bool = False
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool = False
    themes: list[str] | None = None
    no_themes: bool = False
    no_context_files: bool = False
    list_models: str | bool | None = None
    offline: bool = False
    ui_mode: str | None = None
    verbose: bool = False
    project_trust_override: bool = False
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)
    diagnostics: list[dict[str, str]] = field(default_factory=list)


def parse_args(args: Sequence[str]) -> Args:
    """Parse CLI arguments into an Args dataclass."""
    result = Args()
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 1
            if mode in VALID_MODES:
                result.mode = mode
        elif arg in ("--continue", "-c"):
            result.continue_session = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(args):
            i += 1
            result.provider = args[i]
        elif arg == "--model" and i + 1 < len(args):
            i += 1
            result.model = args[i]
        elif arg == "--api-key" and i + 1 < len(args):
            i += 1
            result.api_key = args[i]
        elif arg == "--system-prompt" and i + 1 < len(args):
            i += 1
            result.system_prompt = args[i]
        elif arg == "--append-system-prompt" and i + 1 < len(args):
            i += 1
            if result.append_system_prompt is None:
                result.append_system_prompt = []
            result.append_system_prompt.append(args[i])
        elif arg in ("--name", "-n") and i + 1 < len(args):
            i += 1
            result.name = args[i]
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(args):
            i += 1
            result.session = args[i]
        elif arg == "--session-id" and i + 1 < len(args):
            i += 1
            result.session_id = args[i]
        elif arg == "--fork" and i + 1 < len(args):
            i += 1
            result.fork = args[i]
        elif arg == "--session-dir" and i + 1 < len(args):
            i += 1
            result.session_dir = args[i]
        elif arg == "--models" and i + 1 < len(args):
            i += 1
            result.models = [s.strip() for s in args[i].split(",")]
        elif arg in ("--no-tools", "-nt"):
            result.no_tools = True
        elif arg in ("--no-builtin-tools", "-nbt"):
            result.no_builtin_tools = True
        elif arg in ("--tools", "-t") and i + 1 < len(args):
            i += 1
            result.tools = [s.strip() for s in args[i].split(",") if s.strip()]
        elif arg in ("--exclude-tools", "-xt") and i + 1 < len(args):
            i += 1
            result.exclude_tools = [s.strip() for s in args[i].split(",") if s.strip()]
        elif arg == "--thinking" and i + 1 < len(args):
            i += 1
            level = args[i]
            if is_valid_thinking_level(level):
                result.thinking = level
            else:
                result.diagnostics.append(
                    {
                        "type": "warning",
                        "message": (
                            f'Invalid thinking level "{level}". Valid values: {", ".join(VALID_THINKING_LEVELS)}'
                        ),
                    }
                )
        elif arg in ("--print", "-p"):
            result.print = True
            if i + 1 < len(args):
                next_arg = args[i + 1]
                if not next_arg.startswith("@") and (not next_arg.startswith("-") or next_arg.startswith("---")):
                    result.messages.append(next_arg)
                    i += 1
        elif arg == "--export" and i + 1 < len(args):
            i += 1
            result.export = args[i]
        elif arg in ("--extension", "-e") and i + 1 < len(args):
            i += 1
            if result.extensions is None:
                result.extensions = []
            result.extensions.append(args[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(args):
            i += 1
            if result.skills is None:
                result.skills = []
            result.skills.append(args[i])
        elif arg == "--prompt-template" and i + 1 < len(args):
            i += 1
            if result.prompt_templates is None:
                result.prompt_templates = []
            result.prompt_templates.append(args[i])
        elif arg == "--theme" and i + 1 < len(args):
            i += 1
            if result.themes is None:
                result.themes = []
            result.themes.append(args[i])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg in ("--no-context-files", "-nc"):
            result.no_context_files = True
        elif arg == "--list-models":
            if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@"):
                i += 1
                result.list_models = args[i]
            else:
                result.list_models = True
        elif arg == "--ui-mode" and i + 1 < len(args):
            mode = args[i + 1]
            if mode in VALID_UI_MODES:
                result.ui_mode = mode
                i += 1
            else:
                i += 1
                result.diagnostics.append(
                    {"type": "error", "message": f'Invalid UI mode "{mode}". Valid values: regular, fullscreen'}
                )
        elif arg == "--alt":
            result.ui_mode = "fullscreen"
        elif arg == "--verbose":
            result.verbose = True
        elif arg in ("--approve", "-a"):
            result.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            result.project_trust_override = False
        elif arg == "--offline":
            result.offline = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--"):
            # Unknown flag - could be an extension flag
            if "=" in arg:
                key, value = arg[2:].split("=", 1)
                result.unknown_flags[key] = value
            else:
                result.unknown_flags[arg[2:]] = True
        else:
            result.messages.append(arg)

        i += 1

    return result


HELP_TEXT = f"""\
Pi Coding Agent

Usage: pi [OPTIONS] [PROMPT]

Options:
  -h, --help              Show this help message
  -v, --version           Show version
  -p, --print             Print mode (non-interactive)
      --mode MODE         Output mode: {", ".join(VALID_MODES)}
  -c, --continue          Continue most recent session
  -r, --resume            Resume a session
      --provider NAME     LLM provider
      --model NAME        Model name
      --api-key KEY       API key
      --system-prompt P   Override system prompt
      --thinking LEVEL    Thinking level: {", ".join(VALID_THINKING_LEVELS)}
  -n, --name NAME         Session name
      --no-session        Skip session creation
      --session NAME      Session name
      --session-id ID     Session ID
      --fork ID           Fork from session
      --session-dir DIR   Session directory
      --models LIST       Comma-separated model list
  -t, --tools LIST        Comma-separated tool names
  -xt, --exclude-tools L  Comma-separated tools to exclude
  -nt, --no-tools         Disable all tools
  -nbt, --no-builtin-tools Disable built-in tools
  -e, --extension PATH    Load extension
  -ne, --no-extensions    Disable extensions
      --skill NAME        Load skill
  -ns, --no-skills        Disable skills
      --prompt-template N Load prompt template
  -np, --no-prompt-templates Disable prompt templates
      --theme NAME       Load theme
      --no-themes         Disable themes
  -nc, --no-context-files Disable context files
      --list-models [P]  List available models
      --ui-mode MODE      UI mode: regular, fullscreen
      --alt              Fullscreen alias
      --verbose          Verbose output
  -a, --approve          Approve project trust
  -na, --no-approve      Reject project trust
      --offline          Offline mode
      --export FILE      Export session to file

Arguments:
  PROMPT                  Prompt text (use -p for print mode)
  @FILE                   File argument (prefixed with @)
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.help:
        print(HELP_TEXT)
        return 0

    if args.version:
        print(f"{APP_NAME} v0.83.0")
        return 0

    if args.diagnostics:
        for d in args.diagnostics:
            if d["type"] == "error":
                print(f"Error: {d['message']}", file=sys.stderr)

    if args.print or args.mode in ("json", "rpc"):
        # Print mode: run prompt via the agent loop
        from pi_coding_agent.print_mode import run_print_mode_sync

        return run_print_mode_sync(args)

    # Interactive mode would start the TUI here
    print(f"{APP_NAME} - Interactive mode not yet implemented in Python port")
    return 0


if __name__ == "__main__":
    sys.exit(main())
