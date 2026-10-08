from datetime import datetime


class AlertManager:
    """
    Manages live NIDS alerts.

    One network flow should correspond to one alert.
    Later CNN checkpoints update the same alert instead
    of creating duplicate rows.
    """

    def __init__(self, max_alerts=100):
        self.max_alerts = max_alerts
        self.alerts = []

    # =========================================================
    # CREATE OR UPDATE ALERT
    # =========================================================

    def upsert_alert(
        self,
        class_name,
        confidence,
        severity,
        source_ip=None,
        destination_ip=None,
        source_port=None,
        destination_port=None,
        protocol=None,
        checkpoint=None,
        flow_key=None,
        detection_status="Detected",
    ):
        """
        Create a new alert for a flow, or update the existing
        alert belonging to that same flow.

        This prevents:

            packet 4 -> DDoS -> alert
            packet 9 -> DDoS -> second alert

        Instead:

            packet 4 -> DDoS -> alert created
            packet 9 -> DDoS -> same alert updated
        """

        confidence = float(confidence)

        # -----------------------------------------------------
        # Find existing alert belonging to this flow
        # -----------------------------------------------------

        existing_alert = None

        if flow_key is not None:
            for alert in self.alerts:
                if alert.get("flow_key") == flow_key:
                    existing_alert = alert
                    break

        # -----------------------------------------------------
        # UPDATE EXISTING ALERT
        # -----------------------------------------------------

        if existing_alert is not None:

            existing_alert["attack_type"] = class_name
            existing_alert["confidence"] = round(
                confidence,
                2
            )

            existing_alert["severity"] = severity

            existing_alert["source_ip"] = source_ip
            existing_alert["destination_ip"] = destination_ip

            existing_alert["source_port"] = source_port
            existing_alert["destination_port"] = destination_port

            existing_alert["protocol"] = protocol

            existing_alert["checkpoint"] = checkpoint

            existing_alert["detection_status"] = (
                detection_status
            )

            existing_alert["last_updated"] = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            return existing_alert

        # -----------------------------------------------------
        # CREATE NEW ALERT
        # -----------------------------------------------------

        alert = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "last_updated": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "attack_type": class_name,

            "confidence": round(
                confidence,
                2
            ),

            "severity": severity,

            "source_ip": source_ip,

            "destination_ip": destination_ip,

            "source_port": source_port,

            "destination_port": destination_port,

            "protocol": protocol,

            "checkpoint": checkpoint,

            "detection_status": detection_status,

            "flow_key": flow_key,
        }

        # Newest alert goes first.
        self.alerts.insert(
            0,
            alert
        )

        # Keep memory bounded.
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[
                :self.max_alerts
            ]

        return alert

    # =========================================================
    # BACKWARD-COMPATIBLE CREATE
    # =========================================================

    def create_alert(
        self,
        class_name,
        confidence,
        severity,
        source_ip=None,
        destination_ip=None,
        source_port=None,
        destination_port=None,
        checkpoint=None,
        protocol=None,
        flow_key=None,
        detection_status="Detected",
    ):
        """
        Compatibility wrapper.

        Existing code calling create_alert() will still work.
        """

        return self.upsert_alert(
            class_name=class_name,
            confidence=confidence,
            severity=severity,
            source_ip=source_ip,
            destination_ip=destination_ip,
            source_port=source_port,
            destination_port=destination_port,
            protocol=protocol,
            checkpoint=checkpoint,
            flow_key=flow_key,
            detection_status=detection_status,
        )

    # =========================================================
    # GET ALERTS
    # =========================================================

    def get_alerts(self):
        """
        Return a copy so callers cannot accidentally modify
        the internal alert list.
        """

        return [
            dict(alert)
            for alert in self.alerts
        ]

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):
        self.alerts.clear()