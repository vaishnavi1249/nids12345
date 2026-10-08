import os
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "anomaly_detector.joblib"
)

FEATURE_VERSION = "v1"

# Conservative runtime threshold.
#
# Isolation Forest:
#   higher score = more normal
#   lower score  = more anomalous
#
# The original learned threshold was approximately zero,
# which produced too many live false positives.
DEFAULT_ANOMALY_THRESHOLD = -0.20


class AnomalyDetector:
    """
    Isolation Forest based network anomaly detector.

    The model is trained only on Normal traffic.

    Pipeline:

        CNN
         ↓
        Confidence Filter
         ↓
        Isolation Forest
         ↓
        Normal / Known Attack / Unknown Anomaly
    """

    def __init__(
        self,
        confidence_threshold=75.0,
        model_path=MODEL_PATH
    ):

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.model_path = model_path

        self.model = None

        self.threshold = (
            DEFAULT_ANOMALY_THRESHOLD
        )

        self.feature_version = (
            FEATURE_VERSION
        )

        self.training_samples = 0

        self.model_available = False

        self._load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):

        if not os.path.exists(
            self.model_path
        ):

            print(
                "[!] Anomaly detector model "
                "not found:"
            )

            print(
                f"    {self.model_path}"
            )

            return


        try:

            bundle = joblib.load(
                self.model_path
            )


            # ------------------------------------------------
            # Bundle format
            # ------------------------------------------------

            if isinstance(
                bundle,
                dict
            ):

                self.model = bundle.get(
                    "model"
                )

                self.feature_version = (
                    bundle.get(
                        "feature_version",
                        FEATURE_VERSION
                    )
                )

                self.training_samples = int(
                    bundle.get(
                        "training_samples",
                        0
                    )
                )

                saved_threshold = (
                    bundle.get(
                        "threshold"
                    )
                )


                # --------------------------------------------
                # Runtime threshold
                # --------------------------------------------

                if (
                    saved_threshold is not None
                    and np.isfinite(
                        float(saved_threshold)
                    )
                ):

                    saved_threshold = float(
                        saved_threshold
                    )


                    # The original threshold was
                    # approximately zero.
                    #
                    # Do not use a threshold that
                    # aggressive for live traffic.

                    if saved_threshold <= -0.08:

                        self.threshold = (
                            saved_threshold
                        )

                    else:

                        self.threshold = (
                            DEFAULT_ANOMALY_THRESHOLD
                        )

                else:

                    self.threshold = (
                        DEFAULT_ANOMALY_THRESHOLD
                    )


            # ------------------------------------------------
            # Direct model format
            # ------------------------------------------------

            else:

                self.model = bundle

                self.threshold = (
                    DEFAULT_ANOMALY_THRESHOLD
                )


            if self.model is None:

                raise ValueError(
                    "Isolation Forest model "
                    "was not found in bundle."
                )


            self.model_available = True


            print(
                "[+] Isolation Forest loaded."
            )

            print(
                f"    Feature version: "
                f"{self.feature_version}"
            )

            print(
                f"    Training samples: "
                f"{self.training_samples}"
            )

            print(
                f"    Runtime threshold: "
                f"{self.threshold:.6f}"
            )


        except Exception as e:

            print(
                "[!] Failed to load anomaly "
                f"detector: {e}"
            )

            self.model = None

            self.model_available = False


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    @staticmethod
    def extract_features(
        image
    ):
        """
        Extract the ORIGINAL 31 anomaly features.

        IMPORTANT:
        This exact feature layout must remain compatible
        with the already-trained Isolation Forest.

        Feature groups:

        1. Global channel statistics
           3 channels × 4 = 12

        2. Packet-level statistics
           3 channels × 4 = 12

        3. Temporal packet differences
           3 channels × 2 = 6

        4. Direction balance
           1

        TOTAL = 31 FEATURES
        """

        image = np.asarray(
            image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        if image.ndim == 4:

            image = image[0]


        if image.ndim != 3:

            raise ValueError(
                "Expected image shape "
                "(9, 1486, 3)"
            )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image = image / 255.0


        features = []


        # ====================================================
        # 1. GLOBAL CHANNEL STATISTICS
        # ====================================================
        #
        # 3 channels × 4 statistics = 12
        #
        # mean
        # std
        # min
        # max
        # ====================================================

        for channel in range(3):

            channel_data = (
                image[:, :, channel]
            )


            features.append(
                float(
                    np.mean(
                        channel_data
                    )
                )
            )


            features.append(
                float(
                    np.std(
                        channel_data
                    )
                )
            )


            features.append(
                float(
                    np.min(
                        channel_data
                    )
                )
            )


            features.append(
                float(
                    np.max(
                        channel_data
                    )
                )
            )


        # ====================================================
        # 2. PACKET-LEVEL STATISTICS
        # ====================================================
        #
        # 3 channels × 4 statistics = 12
        #
        # For each RGB channel:
        #
        #   mean packet intensity
        #   std packet intensity
        #   minimum packet intensity
        #   maximum packet intensity
        # ====================================================

        for channel in range(3):

            channel_data = (
                image[:, :, channel]
            )


            packet_means = np.mean(
                channel_data,
                axis=1
            )


            features.append(
                float(
                    np.mean(
                        packet_means
                    )
                )
            )


            features.append(
                float(
                    np.std(
                        packet_means
                    )
                )


            )


            features.append(
                float(
                    np.min(
                        packet_means
                    )
                )
            )


            features.append(
                float(
                    np.max(
                        packet_means
                    )
                )
            )


        # ====================================================
        # 3. TEMPORAL PACKET DIFFERENCES
        # ====================================================
        #
        # Difference between consecutive packets.
        #
        # 3 channels × 2 statistics = 6
        #
        # mean absolute difference
        # standard deviation of absolute difference
        # ====================================================

        if image.shape[0] > 1:

            differences = np.diff(
                image,
                axis=0
            )


            absolute_differences = (
                np.abs(
                    differences
                )
            )


            for channel in range(3):

                channel_diff = (
                    absolute_differences[
                        :,
                        :,
                        channel
                    ]
                )


                features.append(
                    float(
                        np.mean(
                            channel_diff
                        )
                    )
                )


                features.append(
                    float(
                        np.std(
                            channel_diff
                        )
                    )
                )


        else:

            for _ in range(3):

                features.append(0.0)
                features.append(0.0)


        # ====================================================
        # 4. DIRECTION BALANCE
        # ====================================================
        #
        # Red = forward packets
        # Green = backward packets
        #
        # One ratio feature.
        # ====================================================

        forward_activity = float(
            np.mean(
                image[:, :, 0]
            )
        )


        backward_activity = float(
            np.mean(
                image[:, :, 1]
            )
        )


        total_activity = (
            forward_activity
            + backward_activity
        )


        if total_activity > 1e-8:

            direction_ratio = (
                forward_activity
                / total_activity
            )

        else:

            direction_ratio = 0.5


        features.append(
            float(
                direction_ratio
            )
        )


        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        features = np.asarray(
            features,
            dtype=np.float32
        )


        if features.shape[0] != 31:

            raise ValueError(
                "Unexpected anomaly feature "
                f"count: {features.shape[0]}; "
                "expected 31."
            )


        return features


    # ========================================================
    # ANOMALY SCORE
    # ========================================================

    def _get_anomaly_score(
        self,
        features
    ):

        if self.model is None:

            return 0.0


        score = float(
            self.model.decision_function(
                features.reshape(
                    1,
                    -1
                )
            )[0]
        )


        if not np.isfinite(
            score
        ):

            return 0.0


        return score


    # ========================================================
    # SEVERITY
    # ========================================================

    @staticmethod
    def _severity(
        classification
    ):

        if classification in {
            "DDoS",
            "Infiltration",
            "Heartbleed",
        }:

            return "Critical"


        if classification in {
            "DoS",
            "Botnet",
            "Web Attack",
            "Unknown Anomaly",
        }:

            return "High"


        if classification in {
            "Port Scan",
            "Brute Force",
        }:

            return "Medium"


        return "Low"


    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
    self,
    prediction_label,
    confidence_score,
    image=None,
    packet_count=9
    ):
        """
        Analyze CNN prediction + RGB packet image.
        """

        confidence_score = float(
            confidence_score
        )


        if not np.isfinite(
            confidence_score
        ):

            confidence_score = 0.0


        confidence_score = max(
            0.0,
            min(
                confidence_score,
                100.0
            )
        )


        # ====================================================
        # KNOWN CNN ATTACK
        # ====================================================

        if (
            prediction_label != "Normal"
            and confidence_score
            >= self.confidence_threshold
        ):

            severity = (
                self._severity(
                    prediction_label
                )
            )


            return {

                "classification":
                    prediction_label,

                "is_anomaly":
                    True,

                "status":
                    "Known Attack",

                "reason":
                    "CNN classified the "
                    "traffic as a known "
                    "attack with sufficient "
                    "confidence.",

                "severity":
                    severity,

                "anomaly_score":
                    0.0,

                "model_available":
                    self.model_available,

            }


        # ====================================================
        # NO ANOMALY MODEL
        # ====================================================

        if (
            image is None
            or not self.model_available
            or self.model is None
        ):

            return {

                "classification":
                    "Normal",

                "is_anomaly":
                    False,

                "status":
                    "Normal",

                "reason":
                    "No anomaly detected.",

                "severity":
                    "Low",

                "anomaly_score":
                    0.0,

                "model_available":
                    self.model_available,

            }


        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        try:

            features = (
                self.extract_features(
                    image
                )
            )

        except Exception as e:

            print(
                "[!] Anomaly feature "
                f"extraction error: {e}"
            )


            return {

                "classification":
                    "Normal",

                "is_anomaly":
                    False,

                "status":
                    "Normal",

                "reason":
                    "Anomaly analysis "
                    "could not be completed.",

                "severity":
                    "Low",

                "anomaly_score":
                    0.0,

                "model_available":
                    self.model_available,

            }


        # ====================================================
        # ISOLATION FOREST SCORE
        # ====================================================

        # ====================================================
# ISOLATION FOREST SCORE
# ====================================================

        anomaly_score = (
            self._get_anomaly_score(
                features
            )
        )

# ====================================================
# EARLY CHECKPOINT PROTECTION
# ====================================================
#
# The CNN is allowed to classify packets at
# packet 1 and packet 4.
#
# The Isolation Forest is used for final
# unknown-anomaly confirmation at packet 9.
#
# This prevents partially built flows from
# being incorrectly flagged as anomalies.
# ====================================================

        if packet_count < 9:

            return {
                "classification": prediction_label,
                "is_anomaly": prediction_label != "Normal",
                "status": (
                    "Known Attack"
                    if prediction_label != "Normal"
                    else "Monitoring"
                ),
                "reason": (
                    "CNN detected a known attack."
                    if prediction_label != "Normal"
                    else "Monitoring flow until the 9-packet anomaly checkpoint."
                ),
                "severity": (
                    self._severity(prediction_label)
                    if prediction_label != "Normal"
                    else "Low"
                ),
                "anomaly_score": round(
                    anomaly_score,
                    6
                ),
                "model_available": self.model_available,
            }

# ====================================================
# UNKNOWN ANOMALY DECISION
# ====================================================

        is_outlier = (
            anomaly_score
            < self.threshold
        )


        # ====================================================
        # UNKNOWN ANOMALY
        # ====================================================

        if (
            prediction_label == "Normal"
            and is_outlier
        ):

            return {

                "classification":
                    "Unknown Anomaly",

                "is_anomaly":
                    True,

                "status":
                    "Unknown Anomaly",

                "reason":
                    "CNN classified the "
                    "traffic as Normal, but "
                    "the observed traffic "
                    "pattern differs from "
                    "the learned normal "
                    "profile.",

                "severity":
                    "High",

                "anomaly_score":
                    round(
                        anomaly_score,
                        6
                    ),

                "model_available":
                    True,

            }


        # ====================================================
        # NORMAL
        # ====================================================

        return {

            "classification":
                "Normal",

            "is_anomaly":
                False,

            "status":
                "Normal",

            "reason":
                "Traffic matches the "
                "learned normal behavior "
                "profile.",

            "severity":
                "Low",

            "anomaly_score":
                round(
                    anomaly_score,
                    6
                ),

            "model_available":
                True,

        }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )

    print(
        "      ANOMALY DETECTOR SELF TEST"
    )

    print(
        "=========================================="
    )


    detector = AnomalyDetector()


    # --------------------------------------------------------
    # Create a valid dummy RGB image
    # --------------------------------------------------------

    test_image = np.zeros(
        (
            9,
            1486,
            3
        ),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Feature test
    # --------------------------------------------------------

    features = (
        detector.extract_features(
            test_image
        )
    )


    print(
        f"[+] Feature shape: "
        f"{features.shape}"
    )


    print(
        f"[+] Feature count: "
        f"{len(features)}"
    )


    # --------------------------------------------------------
    # Analysis test
    # --------------------------------------------------------

    result = detector.analyze(
        "Normal",
        99.0,
        test_image
    )


    print(
        "[+] Test result:"
    )

    print(
        result
    )


    print(
        "=========================================="
    )

    print(
        "[+] Self-test complete."
    )