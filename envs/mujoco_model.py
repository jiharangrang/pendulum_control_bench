from pathlib import Path

import gymnasium as gym


ENV_ID = "InvertedPendulum-v5"
CUSTOM_XML = Path(__file__).resolve().parent / "assets" / "inverted_pendulum_unlimited_hinge.xml"


def make_inverted_pendulum(render_mode=None, **kwargs):
    make_kwargs = {"xml_file": str(CUSTOM_XML)}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    make_kwargs.update(kwargs)
    return gym.make(ENV_ID, **make_kwargs)
