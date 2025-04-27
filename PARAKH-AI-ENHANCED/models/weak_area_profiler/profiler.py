class WeakAreaProfiler:
    def __init__(self):
        pass  # No model needed, just logic

    def build_weak_profile(self, student_attempts):
        """
        Build weak area profile based on doubt levels.

        Parameters:
        - student_attempts: List of dicts, each with {question, topic, doubt_level}

        Example input:
        [
            {"question": "...", "topic": "Sorting Algorithms", "doubt_level": "high_doubt"},
            {"question": "...", "topic": "Data Structures", "doubt_level": "low_doubt"},
            ...
        ]

        Returns:
        - Dict {topic: worst_doubt_level}
        """

        topic_doubt_levels = {}

        # Priority - high_doubt > low_doubt > no_doubt
        doubt_priority = {"high_doubt": 3, "low_doubt": 2, "no_doubt": 1}

        for attempt in student_attempts:
            topic = attempt["topic"]
            doubt = attempt["doubt_level"]

            if topic not in topic_doubt_levels:
                topic_doubt_levels[topic] = doubt
            else:
                # Keep the higher doubt if multiple attempts
                existing_doubt = topic_doubt_levels[topic]
                if doubt_priority[doubt] > doubt_priority[existing_doubt]:
                    topic_doubt_levels[topic] = doubt

        return topic_doubt_levels
