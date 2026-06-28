"""RAX-native lerobot SO-101 gaze engine CLI.

Drop-in replacement for the ``lerobot-gaze-engine`` console script:
mirrors its argument names so existing shell scripts work unchanged.

Run directly::

    python -m manipulation.arms.lerobot_so101 [flags]

Or register via pyproject.toml console_scripts::

    lerobot-gaze-engine = manipulation.arms.lerobot_so101:main
"""

from manipulation.arms.lerobot_so101.entrypoint import main

__all__ = ["main"]
