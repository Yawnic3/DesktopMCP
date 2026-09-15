import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from mcp import Client, StdioServerParameters
from mcp.types import ImageContent, TextContent


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

SYSTEM_PROMPT = """
You are a fast autonomous desktop agent working through an UNGRADED
practice-question activity using only screenshots and basic mouse/keyboard tools.

PRIMARY GOAL
Complete the entire practice activity from beginning to end with minimal delay.

Keep going even if some answers are wrong or confidence becomes low. Never stop
because accuracy is poor, because a previous answer was incorrect, or because
you are uncertain. For every question, choose the best answer you can determine
from the visible information and continue.

AVAILABLE TOOLS
Use only the basic desktop tools:
- screenshot()
- click()
- drag()
- type_text()
- press_key()
- hotkey()
- move_mouse()
- screen_size()
- mouse_position()

QUESTION WORKFLOW
For each question:

1. Take a screenshot of the current screen.

2. Read the full visible question and every visible answer choice.

3. Determine the best answer internally.

4. Immediately select the best answer using click() or drag() as appropriate.

5. Do not waste time explaining routine answers before acting.

6. Confidence does NOT determine whether you answer.
   Always choose the best available answer even if confidence is low.

7. If the page contains High / Medium / Low confidence controls, use:
      High   = confidence >= 90%
      Medium = confidence 70-89%
      Low    = confidence < 70%

8. High / Medium / Low controls are part of the normal question flow and are
   explicitly authorized. Click the appropriate one.

9. If the page shows Next Question, Next, Continue, Check My Work, or another
   obvious per-question progression control, use it.

10. Continue until the page clearly indicates the entire practice activity
    is finished.

SPEED RULES
- Minimize API calls and tool calls.
- Prefer one screenshot per question.
- Do not call move_mouse() before click() unless actually necessary.
- Click directly at the intended coordinates.
- If click() returns success=true, do not take another screenshot solely to
  verify that routine click.
- Do not generate explanations for routine answers.
- After navigation or a page change, take a fresh screenshot because coordinates
  may have changed.
- Prefer accuracy and direct action over unnecessary narration.

MULTIPLE CHOICE
For a normal multiple-choice question, aim for:

    screenshot
    -> determine answer
    -> click answer
    -> click confidence control
    -> advance
    -> screenshot next question

DRAG-AND-DROP / MATCHING
- Use drag() for matching and drag-and-drop questions.
- Identify the center of the source tile and the center of its destination.
- Drag center-to-center.
- Complete all obvious matches before requesting another screenshot.
- After finishing the entire matching question, verify the final arrangement once.
- If an item snaps back or lands incorrectly, take another screenshot and retry
  with better coordinates.

INCORRECT ANSWERS
- A wrong answer is NEVER a reason to stop.
- If the site reports the previous answer was incorrect:
    inspect visible feedback if useful
    learn from it if useful
    continue to the next question
- Confidence applies only to the current question.
- Even if several answers in a row are wrong, KEEP GOING.

LOW CONFIDENCE
- Never stop solely because confidence is low.
- Never ask the user to answer merely because you are uncertain.
- Pick the best available answer anyway.
- Use Medium or Low when appropriate and continue.
- If two options seem similarly plausible, choose the one best supported by the
  visible wording and proceed.

CONTINUATION RULES
- Do NOT stop merely because the next question is not immediately visible.
- After completing a question, look for:
    High
    Medium
    Low
    Next Question
    Next
    Continue
    Check My Work
    Retry
    another obvious progression control
- A loading screen, transition, animation, or temporary blank state is NOT completion.
- If no next question appears immediately, inspect the screen again.
- If uncertain whether the activity is finished, take another screenshot instead
  of stopping.
- Keep working regardless of previous accuracy.

TOOL SUCCESS / FAILURE
- Never claim an action succeeded unless the tool reports success=true or the
  visible screen clearly confirms it.
- If click() or drag() fails:
    take another screenshot
    reassess the coordinates
    retry
- Do not abandon the activity because one interaction fails.
- Do not repeatedly click the same failed coordinates without re-inspecting the screen.

PAGE CHANGES
- If the page changes unexpectedly, take another screenshot before acting.
- Do not rely on stale coordinates after navigation.
- Be careful with popups, overlays, and browser windows that may block the target.

COMPLETION
Only stop automatically when the page clearly indicates that the entire practice
activity is finished and there is no next practice question or progression control.

Do NOT stop because:
- confidence is low
- an answer was wrong
- several answers were wrong
- a click failed
- a drag failed
- a screenshot was temporarily unclear
- the next question took time to load

FINAL SUBMISSION
You may automatically complete and advance individual practice questions.

You may click:
- High
- Medium
- Low
- Next Question
- Next
- Continue
- Check My Work
- similar per-question controls

Do NOT click a final course-level control such as:
- Submit Assignment
- Finish Attempt
- Turn In
- Submit Entire Assignment

unless the user explicitly requests final submission during the live session.
"""

def mcp_tools_to_openai(mcp_tools):
    tools = []

    for tool in mcp_tools:
        schema = tool.input_schema or {"type": "object", "properties": {}}

        tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": schema,
            }
        )

    return tools


def unpack_mcp_result(result):
    """
    Convert an MCP CallToolResult into:
      - a short text result for function_call_output
      - zero or more image data URLs for the model to inspect
    """
    text_parts = []
    image_urls = []

    for block in result.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)

        elif isinstance(block, ImageContent):
            mime_type = getattr(block, "mime_type", None) or "image/png"

            # MCP ImageContent.data is already a base64 string.
            image_urls.append(f"data:{mime_type};base64,{block.data}")

        else:
            text_parts.append(str(block))

    if getattr(result, "structured_content", None):
        try:
            text_parts.append(
                json.dumps(result.structured_content, ensure_ascii=False)
            )
        except TypeError:
            text_parts.append(str(result.structured_content))

    if getattr(result, "is_error", False):
        text_parts.insert(0, "MCP tool returned an error.")

    if image_urls and not text_parts:
        text_parts.append(
            f"Tool succeeded and returned {len(image_urls)} screenshot image(s)."
        )

    return "\n".join(text_parts).strip() or "Tool completed successfully.", image_urls


async def run_agent(task: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            'PowerShell example:\n'
            '$env:OPENAI_API_KEY="your_api_key_here"'
        )

    server_path = Path(__file__).with_name("server.py")

    if not server_path.exists():
        raise FileNotFoundError(
            f"Could not find server.py next to agent.py:\n{server_path}"
        )

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(server_path.parent),
    )

    openai = AsyncOpenAI()

    print(f"Model: {MODEL}")
    print(f"MCP server: {server_path}")
    print("Connecting to desktop MCP server...")

    async with Client(server) as mcp:
        listed = await mcp.list_tools()
        openai_tools = mcp_tools_to_openai(listed.tools)

        print(
            "Connected. MCP tools:",
            ", ".join(tool.name for tool in listed.tools),
        )
        print("\nStarting agent. Ctrl+C stops it.\n")

        response = await openai.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": task,
                        }
                    ],
                }
            ],
            tools=openai_tools,
        )

        while True:
            if response.output_text:
                print("\nAGENT:")
                print(response.output_text)

            calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if not calls:
                print("\nAgent stopped: no more tool calls.")
                return

            next_input = []

            for call in calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as exc:
                    next_input.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": f"Invalid JSON arguments: {exc}",
                        }
                    )
                    continue

                print(f"\nTOOL: {call.name}({json.dumps(arguments)})")

                try:
                    result = await mcp.call_tool(call.name, arguments)
                    text_result, image_urls = unpack_mcp_result(result)
                except Exception as exc:
                    text_result = f"Tool execution failed: {type(exc).__name__}: {exc}"
                    image_urls = []

                print(f"RESULT: {text_result[:500]}")

                # Every function call must receive a matching function_call_output.
                next_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": text_result,
                    }
                )

                # If the MCP tool returned an image (normally screenshot), attach it
                # as a fresh user message so the vision model can inspect it.
                for image_url in image_urls:
                    next_input.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"This image is the visual result returned by "
                                        f"the MCP tool '{call.name}'. Inspect it before "
                                        f"deciding the next desktop action."
                                    ),
                                },
                                {
                                    "type": "input_image",
                                    "image_url": image_url,
                                },
                            ],
                        }
                    )

            response = await openai.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                reasoning={"effort": "none"},
                text={"verbosity": "low"},
                previous_response_id=response.id,
                input=next_input,
                tools=openai_tools,
            )


async def main():
    default_task = (
        "Look at the current browser screen and work through the ungraded "
        "multiple-choice practice questions. For answers where you are at least "
        "90% confident, select the choice and continue. If confidence is below "
        "90%, stop and ask me. Never click a final Submit/Finish/Turn In control."
    )

    print("Desktop MCP Agent")
    print("-----------------")
    entered = input(
        "Task (press Enter for the default practice-question task):\n> "
    ).strip()

    task = entered or default_task

    await run_agent(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise
