"""
Analytics generation for Station Concordia simulations.

Generates detailed analytics reports for:
- Route changes (blocked exits, re-routing)
- Waiting behavior (information seeking, reasons)
- Messaging (communication patterns, recipients)
"""

from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyticsGenerator:
    """
    Generates analytics reports from simulation data.

    This class takes simulation results and generates human-readable
    analytics files for various behavioral patterns.
    """

    @staticmethod
    def save_all_analytics(
        output_path: Path,
        route_changes: list[dict[str, Any]],
        wait_events: list[dict[str, Any]],
        message_history: list[dict[str, Any]],
    ) -> None:
        """
        Save all analytics reports to files.

        Args:
            output_path: Base output path (analytics saved to same directory)
            route_changes: List of route change events
            wait_events: List of waiting behavior events
            message_history: List of all messages sent
        """
        AnalyticsGenerator._save_route_change_analytics(output_path, route_changes)
        AnalyticsGenerator._save_wait_behavior_analytics(output_path, wait_events)
        AnalyticsGenerator._save_message_analytics(output_path, message_history)

    @staticmethod
    def _save_route_change_analytics(
        output_path: Path, route_changes: list[dict[str, Any]]
    ) -> None:
        """Save route change analytics."""
        if not route_changes:
            return

        route_changes_path = output_path.parent / "route_changes.txt"
        with open(route_changes_path, "w") as f:
            f.write("=== ROUTE CHANGE ANALYTICS ===\n\n")
            f.write(f"Total route changes: {len(route_changes)}\n")
            f.write(f"Agents who changed routes: {len({rc['agent'] for rc in route_changes})}\n\n")
            f.write("Route Changes:\n")
            for rc in route_changes:
                f.write(
                    f"  - {rc['agent']} at t={rc['time']:.1f}s: "
                    f"{rc['from_exit']} → {rc['to_exit']}\n"
                    f"    Reason: {rc['reason']}\n"
                )
        logger.info(f"Route change analytics saved to {route_changes_path}")

    @staticmethod
    def _save_wait_behavior_analytics(output_path: Path, wait_events: list[dict[str, Any]]) -> None:
        """Save waiting behavior analytics."""
        if not wait_events:
            return

        wait_analytics_path = output_path.parent / "wait_behavior.txt"
        with open(wait_analytics_path, "w") as f:
            f.write("=== WAITING BEHAVIOR ANALYTICS ===\n\n")
            f.write(f"Total wait events: {len(wait_events)}\n")
            f.write(f"Agents who waited: {len({w['agent'] for w in wait_events})}\n\n")

            # Breakdown by wait reason
            reason_counts = {}
            for event in wait_events:
                reason = event.get("wait_reason", "unspecified")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            f.write("Wait events by reason:\n")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(wait_events)) * 100 if wait_events else 0
                f.write(f"  {reason}: {count} waits ({percentage:.1f}%)\n")

            # Breakdown by personality type
            personality_counts = {}
            for event in wait_events:
                personality = event.get("personality", "UNKNOWN")
                personality_counts[personality] = personality_counts.get(personality, 0) + 1

            f.write("\nWait events by personality:\n")
            for personality, count in sorted(
                personality_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / len(wait_events)) * 100 if wait_events else 0
                f.write(f"  {personality}: {count} waits ({percentage:.1f}%)\n")

            f.write("\nRecent Wait Events (last 20):\n")
            for event in wait_events[-20:]:
                reason = event.get("wait_reason", "unspecified")
                f.write(
                    f"  - t={event['time']:.1f}s: {event['agent']} ({event['personality']}) "
                    f"waited ({reason})\n"
                )
        logger.info(f"Wait behavior analytics saved to {wait_analytics_path}")

    @staticmethod
    def _save_message_analytics(output_path: Path, message_history: list[dict[str, Any]]) -> None:
        """Save messaging analytics."""
        if not message_history:
            return

        message_analytics_path = output_path.parent / "message_analytics.txt"
        with open(message_analytics_path, "w") as f:
            f.write("=== MESSAGING ANALYTICS ===\n\n")
            f.write(f"Total messages sent: {len(message_history)}\n")
            f.write(f"Agents who sent messages: {len({m['sender'] for m in message_history})}\n")
            total_recipients = sum(m["num_recipients"] for m in message_history)
            f.write(f"Total message deliveries: {total_recipients}\n")
            if message_history:
                avg_recipients = total_recipients / len(message_history)
                f.write(f"Average recipients per message: {avg_recipients:.1f}\n\n")

            # Breakdown by sender
            sender_counts = {}
            for msg in message_history:
                sender = msg["sender"]
                sender_counts[sender] = sender_counts.get(sender, 0) + 1

            f.write("Messages by sender:\n")
            for sender, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {sender}: {count} messages\n")

            # Breakdown by message type
            type_counts = {}
            for msg in message_history:
                msg_type = msg.get("message_type", "broadcast")
                type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

            f.write("\nMessages by type:\n")
            for msg_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(message_history)) * 100
                f.write(f"  {msg_type}: {count} messages ({percentage:.1f}%)\n")

            # Count targeted vs broadcast
            targeted = sum(1 for m in message_history if m.get("target_agent"))
            broadcast = len(message_history) - targeted
            f.write(f"\nTargeted messages: {targeted} ({targeted/len(message_history)*100:.1f}%)\n")
            f.write(
                f"Broadcast messages: {broadcast} ({broadcast/len(message_history)*100:.1f}%)\n"
            )

            f.write("\nAll Messages:\n")
            for msg in message_history:
                msg_type = msg.get("message_type", "broadcast")
                target = msg.get("target_agent")
                type_indicator = f"[{msg_type}]" if msg_type != "broadcast" else ""
                target_indicator = f" → {target}" if target and target != "null" else ""
                f.write(
                    f"  - t={msg['time']:.1f}s: {msg['sender']}{target_indicator} {type_indicator} to {msg['num_recipients']} "
                    f"people: \"{msg['text']}\"\n"
                )
        logger.info(f"Message analytics saved to {message_analytics_path}")
