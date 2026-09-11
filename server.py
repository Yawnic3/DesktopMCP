from io import BytesIO
from typing import Literal

import pyautogui

from mcp.server import MCPServer
from mcp.server.mcpserver import Image


mcp = MCPServer("Local Desktop Controller")

# Safety features built into PyAutoGUI.
# Moving the mouse into the top-left corner aborts PyAutoGUI operations.
pyautogui.FAILSAFE = True

# Small delay after each PyAutoGUI action.
pyautogui.PAUSE = 0.15


def approve(message: str) -> bool:
    """
    Require a real local user confirmation before an action occurs.

    This uses a GUI dialog instead of stdin, because stdin is being used
    by the MCP stdio transport.
    """
    result = pyautogui.confirm(
        text=message,
        title="Desktop MCP approval",
        buttons=["Approve", "Cancel"],
    )

    return result == "Approve"


@mcp.tool()
def screen_size() -> dict[str, int]:
    """Return the current desktop width and height in pixels."""
    width, height = pyautogui.size()

    return {
        "width": width,
        "height": height,
    }


@mcp.tool()
def mouse_position() -> dict[str, int]:
    """Return the mouse cursor's current screen coordinates."""
    x, y = pyautogui.position()

    return {
        "x": x,
        "y": y,
    }


@mcp.tool()
def screenshot() -> Image:
    """
    Take a screenshot of the desktop.

    Use this to visually understand what is currently on screen
    before deciding where to move or click.
    """
    img = pyautogui.screenshot()

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return Image(
        data=buffer.getvalue(),
        format="png",
    )


@mcp.tool()
def move_mouse(
    x: int,
    y: int,
    duration: float = 0.2,
) -> dict:
    """Move the mouse cursor to an absolute screen coordinate."""

    width, height = pyautogui.size()

    if not (0 <= x < width):
        raise ValueError(f"x must be between 0 and {width - 1}")

    if not (0 <= y < height):
        raise ValueError(f"y must be between 0 and {height - 1}")

    duration = max(0.0, min(duration, 3.0))

    pyautogui.moveTo(
        x,
        y,
        duration=duration,
    )

    return {
        "success": True,
        "x": x,
        "y": y,
    }


@mcp.tool()
def click(
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
    clicks: int = 1,
) -> dict:
    """
    Click a desktop coordinate.

    The local user must approve the click through a desktop dialog.
    """

    width, height = pyautogui.size()

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Coordinates are outside the screen.")

    clicks = max(1, min(clicks, 3))

    approved = approve(
        f"Allow agent to {button}-click at ({x}, {y}) "
        f"{clicks} time(s)?"
    )

    if not approved:
        return {
            "success": False,
            "reason": "User cancelled action.",
        }

    pyautogui.click(
        x=x,
        y=y,
        clicks=clicks,
        button=button,
        interval=0.1,
    )

    return {
        "success": True,
        "x": x,
        "y": y,
        "clicks": clicks,
        "button": button,
    }


@mcp.tool()
def type_text(
    text: str,
    interval: float = 0.02,
) -> dict:
    """
    Type text into the currently focused field.

    Requires local user approval.
    """

    if len(text) > 2000:
        raise ValueError("Refusing to type more than 2000 characters at once.")

    preview = text

    if len(preview) > 200:
        preview = preview[:200] + "..."

    approved = approve(
        f"The agent wants to type:\n\n{preview}\n\nAllow?"
    )

    if not approved:
        return {
            "success": False,
            "reason": "User cancelled action.",
        }

    interval = max(0.0, min(interval, 0.5))

    pyautogui.write(
        text,
        interval=interval,
    )

    return {
        "success": True,
        "characters_typed": len(text),
    }


@mcp.tool()
def press_key(
    key: str,
    presses: int = 1,
) -> dict:
    """Press a keyboard key after user approval."""

    presses = max(1, min(presses, 10))

    approved = approve(
        f"Allow agent to press '{key}' {presses} time(s)?"
    )

    if not approved:
        return {
            "success": False,
            "reason": "User cancelled action.",
        }

    pyautogui.press(
        key,
        presses=presses,
        interval=0.1,
    )

    return {
        "success": True,
        "key": key,
        "presses": presses,
    }


@mcp.tool()
def hotkey(
    key1: str,
    key2: str,
    key3: str | None = None,
) -> dict:
    """Execute a keyboard shortcut after user approval."""

    keys = [key1, key2]

    if key3:
        keys.append(key3)

    description = " + ".join(keys)

    approved = approve(
        f"Allow agent to press hotkey:\n\n{description}"
    )

    if not approved:
        return {
            "success": False,
            "reason": "User cancelled action.",
        }

    pyautogui.hotkey(*keys)

    return {
        "success": True,
        "keys": keys,
    }

@mcp.prompt()
def practice_questions() -> str:
    """Instructions for working through multiple-choice practice questions."""
    return """
You are controlling my browser through desktop MCP tools.

Work through the multiple-choice practice questions currently displayed.

For each question:
1. Take a screenshot and read the entire question and every answer choice.
2. Solve the question carefully before interacting with the page.
3. State the answer, a short explanation, and a confidence score from 0-100%.
4. If confidence is at least 90%, click the answer choice.
5. Take another screenshot and verify the intended choice was selected.
6. If confidence is below 90%, stop and ask me instead of guessing.
7. If there is an obvious Next button, continue to the next question.

Never infer an answer solely from its position or highlighting.
If the screen is unclear, take another screenshot instead of guessing.
Do not interact with unrelated tabs or applications.
Do not click a final Submit/Finish button unless I explicitly request it.
"""

if __name__ == "__main__":
    mcp.run()