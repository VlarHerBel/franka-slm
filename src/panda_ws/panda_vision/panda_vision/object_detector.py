#!/usr/bin/env python3
"""Compatibility shim for the legacy object_detector entrypoint."""

from panda_vision.nodes.perception_node import main as perception_main


def main(args=None):
    """Run the active modular perception node."""
    perception_main(args)


if __name__ == "__main__":
    main()
