"""Decision-making: from what an agent perceives to what it does.

==========================  =================================================
``decision_processor``      The decision cycle: who decides, then decide all
                            and apply in a fixed order.
``situation``               Perceive and frame: the :class:`DecisionContext`.
``payload``                 The decision every engine returns; validation and
                            the safe fallback.
``rule_based_decision_engine``  Cue-driven stages and exit scoring.
``llm_decision_engine``     Prompting a language model, with repair, cache and
                            fallback.
``llm_prompt``              Rendering the context into the prompt.
``prompt_cache``            Reusing a decision when nothing significant changed.
``action_executor``         Applying a decision in the pedestrian simulation.
``transfer_routing``        Agents stepping off an escalator.
==========================  =================================================
"""
