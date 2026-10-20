"""
Stat Summary Metric Cards Widget with Cyber-Forensic KPI Cards
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt


class StatCard(QFrame):
    """Individual Metric KPI Card."""

    def __init__(self, title: str, value: str = "0", accent_color: str = "#38bdf8", parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setStyleSheet(f"""
            #StatCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #162238, stop:1 #0f182a);
                border: 1px solid #23334d;
                border-top: 3px solid {accent_color};
                border-radius: 8px;
                padding: 6px 10px;
            }}
            #StatCard:hover {{
                border-color: {accent_color};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.title_label = QLabel(title.upper(), self)
        self.title_label.setObjectName("StatCardTitle")
        self.title_label.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700; letter-spacing: 0.3px;")

        self.value_label = QLabel(value, self)
        self.value_label.setObjectName("StatCardValue")
        self.value_label.setStyleSheet(f"color: {accent_color}; font-size: 19px; font-weight: 800;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(str(value))


class StatSummaryRow(QWidget):
    """Horizontal row of key performance summary cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.card_total = StatCard("Total Frames", "0", "#f8fafc", self)
        self.card_valid = StatCard("Valid Frames", "0", "#10b981", self)
        self.card_injections = StatCard("Injections", "0", "#f43f5e", self)
        self.card_replays = StatCard("Replays", "0", "#fbbf24", self)
        self.card_commands = StatCard("Commands", "0", "#38bdf8", self)
        self.card_anomalies = StatCard("ΔSeq Anom", "0", "#ec4899", self)
        self.card_bad_data = StatCard("Bad Data", "0", "#ef4444", self)
        self.card_mean_dt = StatCard("Mean Δt (ms)", "0.0", "#06b6d4", self)

        layout.addWidget(self.card_total)
        layout.addWidget(self.card_valid)
        layout.addWidget(self.card_injections)
        layout.addWidget(self.card_replays)
        layout.addWidget(self.card_commands)
        layout.addWidget(self.card_anomalies)
        layout.addWidget(self.card_bad_data)
        layout.addWidget(self.card_mean_dt)

    def update_stats(self, stats: dict):
        """Update values from dictionary."""
        if "total_packets" in stats:
            self.card_total.set_value(f"{stats['total_packets']:,}")
        if "mavlink_valid" in stats:
            self.card_valid.set_value(f"{stats['mavlink_valid']:,}")
        if "label_injections" in stats:
            self.card_injections.set_value(f"{stats['label_injections']:,}")
        if "label_replays" in stats:
            self.card_replays.set_value(f"{stats['label_replays']:,}")
        if "commands_count" in stats:
            self.card_commands.set_value(f"{stats['commands_count']:,}")
        if "anomalies_count" in stats:
            self.card_anomalies.set_value(f"{stats['anomalies_count']:,}")
        if "corrupted" in stats:
            self.card_bad_data.set_value(f"{stats['corrupted']:,}")
        if "mean_delta_t_ms" in stats:
            self.card_mean_dt.set_value(f"{stats['mean_delta_t_ms']:.1f}")

    def reset(self):
        self.card_total.set_value("0")
        self.card_valid.set_value("0")
        self.card_injections.set_value("0")
        self.card_replays.set_value("0")
        self.card_commands.set_value("0")
        self.card_anomalies.set_value("0")
        self.card_bad_data.set_value("0")
        self.card_mean_dt.set_value("0.0")
