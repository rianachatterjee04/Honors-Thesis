#!/usr/bin/env python
import sys
import warnings

from physical_agent.crew import PhysicalAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """
    inputs = {
        "start_spot": "pickup location A",
        "handoff_spot": "handoff zone B",
        "target_label": "delivery point C",
    }

    try:
        result = PhysicalAgent().crew().kickoff(inputs=inputs)
        print("Crew execution completed successfully!")
        print(f"Result: {result}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "start_spot": "pickup location A",
        "handoff_spot": "handoff zone B",
        "target_label": "delivery point C",
    }
    try:
        PhysicalAgent().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PhysicalAgent().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "start_spot": "pickup location A",
        "handoff_spot": "handoff zone B",
        "target_label": "delivery point C",
    }

    try:
        PhysicalAgent().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()
