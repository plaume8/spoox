import subprocess
import time


class TmuxTerminalSession:
    """
    Control a persistent tmux pane from Python.

    - A tmux session is created if it doesn't exist.
    - One can send commands / key-presses.
    - One can read the current visible screen contents.
    """

    def __init__(self, session_name="terminal", window_index=0, pane_index=0):
        self.session_name = session_name
        self.window_index = window_index
        self.pane_index = pane_index
        self.target = f"{session_name}:{window_index}.{pane_index}"
        self._ensure_session()

    def _tmux(self, *args, capture_output=True, text=True, check=True):
        """Run a tmux command and return the CompletedProcess."""
        return subprocess.run(
            ["tmux", *args],
            capture_output=capture_output,
            text=text,
            check=check,
        )

    def _ensure_session(self):
        """Create the session if it does not exist yet."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            # Create a detached session running your shell (default: $SHELL)
            self._tmux("new-session", "-d", "-s", self.session_name)
            time.sleep(2)  # todo can be done better
            # set a fixed terminal size
            self._tmux("resize-window", "-t", self.session_name, "-x", "128", "-y", "16")

    def send_line(self, line: str):
        """Send a command followed by Enter (like typing a command and pressing Return)."""
        self._tmux("send-keys", "-t", self.target, line, "Enter")

    def send_keys(self, *keys: str):
        """Send arbitrary keys (no implicit Enter). You can pass tmux key tokens (e.g. 'C-c') or literal strings."""
        self._tmux("send-keys", "-t", self.target, *keys)

    def get_screen(self) -> (str, str):
        """
        Get the *current visible* screen contents of the pane as text and the currently *running program*.

        If you want escape sequences (colors, cursor pos, etc),
        call capture-pane with -e instead and pipe through a terminal emulator.
        """
        # Get the current command
        cmd_proc = self._tmux(
            "display-message",
            "-p",
            "-F",
            "#{pane_current_command}",
            "-t",
            self.target,
        )
        current_cmd = cmd_proc.stdout.strip()

        # Ask tmux for the pane height
        height_proc = self._tmux(
            "display-message",
            "-p",
            "-F",
            "#{pane_height}",
            "-t",
            self.target,
        )
        height = int(height_proc.stdout.strip())

        # Capture the last <height> lines (i.e. full visible screen)
        cap = self._tmux(
            "capture-pane",
            "-t",
            self.target,
            "-p",  # print to stdout
            "-S", f"-{height}",
        )
        screen_text = cap.stdout.rstrip("\n")

        return screen_text, current_cmd

    def clear_history(self):
        """Clear terminal screen history."""
        self.send_line("clear")
        time.sleep(1)  # todo can be done better
        self._tmux("clear-history", "-t", self.target)

    def kill_session(self):
        """Kill the whole tmux session (optional)."""
        self._tmux("kill-session", "-t", self.session_name)
