"""Escalators modelled as two-lane conveyors, detached from the floor simulation.

A real escalator is not floor: once a passenger steps on, the steps carry them
at belt speed regardless of crowding, at most one person per step per lane,
standers on the right and walkers on the left. The only place a queue can form
is at the boarding comb, on the open landing.

This package models exactly that. Riders leave JuPedSim at the boarding comb
(``system.EscalatorSystem``), travel on a ``conveyor.Conveyor`` and are placed
back onto the floor of the other level at the far comb.
"""

from evacusim.escalators.conveyor import Conveyor, ConveyorParams, Rider

__all__ = ["Conveyor", "ConveyorParams", "Rider"]
