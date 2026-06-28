"""Exchange — the agent <-> robot hub (the "X" in RAX).

The exchange brokers the bidirectional stream between robots and agents:
robots publish perception (video, depth, audio); the server runs the heavy
models + agent and exchanges commands and speech back. This is where any
robot platform and any agent meet over a common WebSocket transport.

Contents:
    server.py  Central hub: models + agent + dispatcher + web UI.
"""
