"""
LSTM fault detector integration for live simulation telemetry.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import joblib
import numpy as np

from backend.utils.config import LSTM_MODEL_CONFIG
from backend.utils.logger import LoggerMixin


class LSTMFaultDetector(LoggerMixin):
    """Builds engineered LSTM features from live reactor state."""

    class_names = ["Normal", "Scram", "LOFA"]

    def __init__(self, model_path: Path, config_path: Path, scaler_path: Path) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.scaler_path = Path(scaler_path)

        self.model = None
        self.scaler = None
        self.config: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.raw_history: Deque[Dict[str, float]] = deque(maxlen=8)
        self.feature_buffer: Optional[Deque[np.ndarray]] = None
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.confidence_thresholds = {
            "high_confidence": 0.9,
            "medium_confidence": 0.7,
            "low_confidence": 0.5,
        }

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        missing = [
            str(path)
            for path in (self.model_path, self.config_path, self.scaler_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing LSTM artifacts: {missing}")

        with open(self.config_path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)

        self.config = config_data["config"]
        self.feature_names = list(config_data["feature_names"])
        self.scaler = joblib.load(self.scaler_path)
        self.feature_buffer = deque(maxlen=int(self.config["window_size"]))
        self.model = self._build_inference_model()
        self.model.load_weights(self.model_path)

    def _build_inference_model(self):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            LSTM,
            BatchNormalization,
            Bidirectional,
            Dense,
            Dropout,
            Input,
            Lambda,
            Multiply,
            Softmax,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l1_l2

        input_shape = (int(self.config["window_size"]), len(self.feature_names))
        inputs = Input(shape=input_shape)

        if self.config.get("use_bidirectional", False):
            x = Bidirectional(
                LSTM(
                    self.config["lstm_units"][0],
                    return_sequences=True,
                    dropout=self.config["dropout_rate"],
                    recurrent_dropout=self.config["dropout_rate"],
                    kernel_regularizer=l1_l2(self.config["l1_reg"], self.config["l2_reg"]),
                )
            )(inputs)
        else:
            x = LSTM(
                self.config["lstm_units"][0],
                return_sequences=True,
                dropout=self.config["dropout_rate"],
                recurrent_dropout=self.config["dropout_rate"],
                kernel_regularizer=l1_l2(self.config["l1_reg"], self.config["l2_reg"]),
            )(inputs)

        x = BatchNormalization()(x)

        x = LSTM(
            self.config["lstm_units"][1],
            return_sequences=bool(self.config.get("use_attention", False)),
            dropout=self.config["dropout_rate"],
            recurrent_dropout=self.config["dropout_rate"],
            kernel_regularizer=l1_l2(self.config["l1_reg"], self.config["l2_reg"]),
        )(x)
        x = BatchNormalization()(x)

        if self.config.get("use_attention", False):
            attention_scores = Dense(1, activation="tanh")(x)
            attention_weights = Softmax(axis=1)(attention_scores)
            weighted_sequence = Multiply()([x, attention_weights])
            x = Lambda(
                lambda tensor: tf.reduce_sum(tensor, axis=1),
                output_shape=lambda shape: (shape[0], shape[2]),
                name="attention_reduce_sum",
            )(weighted_sequence)

        for units in self.config["dense_units"]:
            x = Dense(
                units,
                activation="relu",
                kernel_regularizer=l1_l2(self.config["l1_reg"], self.config["l2_reg"]),
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config["dropout_rate"])(x)

        outputs = Dense(3, activation="softmax")(x)
        return Model(inputs=inputs, outputs=outputs)

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.raw_history.clear()
        if self.feature_buffer is None:
            self.feature_buffer = deque(maxlen=int(self.config["window_size"]))
        else:
            self.feature_buffer.clear()

        self.last_prediction = {
            "status": "insufficient_data",
            "message": f"Need {self.config['window_size']} samples before prediction",
        }

        if initial_state is not None:
            return self.update(initial_state)
        return self.last_prediction

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sample = self._extract_raw_sample(state)
        feature_row = self._build_feature_row(sample)
        self.feature_buffer.append(feature_row)
        self.last_prediction = self._predict()
        return self.last_prediction

    def _extract_raw_sample(self, state: Dict[str, Any]) -> Dict[str, float]:
        required = [
            "power",
            "fuel_temp",
            "coolant_temp",
            "pressure",
            "coolant_flow_actual",
            "time",
        ]
        missing = [field for field in required if field not in state]
        if missing:
            raise ValueError(f"Missing reactor fields for LSTM input: {missing}")

        return {
            "Power": float(state["power"]),
            "Fuel_Temp": float(state["fuel_temp"]),
            "Coolant_Temp": float(state["coolant_temp"]),
            "Pressure": float(state["pressure"]),
            "Flow": float(state["coolant_flow_actual"]),
            "Time": float(state["time"]),
        }

    def _build_feature_row(self, sample: Dict[str, float]) -> np.ndarray:
        previous = self.raw_history[-1] if self.raw_history else None
        dt = sample["Time"] - previous["Time"] if previous else 0.1
        dt = dt if dt > 0 else 0.1

        self.raw_history.append(sample)

        power_series = self._series("Power")
        fuel_series = self._series("Fuel_Temp")
        coolant_series = self._series("Coolant_Temp")
        flow_series = self._series("Flow")
        temp_diff_series = [fuel - coolant for fuel, coolant in zip(fuel_series, coolant_series)]

        feature_map = {
            "Power": sample["Power"],
            "Fuel_Temp": sample["Fuel_Temp"],
            "Coolant_Temp": sample["Coolant_Temp"],
            "Pressure": sample["Pressure"],
            "Flow": sample["Flow"],
            "Power_ROC": self._rate(sample["Power"], previous["Power"], dt) if previous else 0.0,
            "Temp_Fuel_ROC": self._rate(sample["Fuel_Temp"], previous["Fuel_Temp"], dt) if previous else 0.0,
            "Temp_Coolant_ROC": self._rate(sample["Coolant_Temp"], previous["Coolant_Temp"], dt) if previous else 0.0,
            "Flow_ROC": self._rate(sample["Flow"], previous["Flow"], dt) if previous else 0.0,
            "Power_MA3": self._rolling_mean(power_series, 3),
            "Power_MA7": self._rolling_mean(power_series, 7),
            "Fuel_Temp_MA3": self._rolling_mean(fuel_series, 3),
            "Fuel_Temp_MA7": self._rolling_mean(fuel_series, 7),
            "Coolant_Temp_MA3": self._rolling_mean(coolant_series, 3),
            "Coolant_Temp_MA7": self._rolling_mean(coolant_series, 7),
            "Power_STD": self._rolling_std(power_series, 5),
            "Fuel_Temp_STD": self._rolling_std(fuel_series, 5),
            "Coolant_Temp_STD": self._rolling_std(coolant_series, 5),
            "Temp_Differential": sample["Fuel_Temp"] - sample["Coolant_Temp"],
            "Temp_Diff_ROC": self._temp_diff_rate(temp_diff_series, dt),
            "Power_Flow_Ratio": sample["Power"] / (sample["Flow"] + 1e-6),
        }

        return np.array([feature_map[name] for name in self.feature_names], dtype=np.float32)

    def _predict(self) -> Dict[str, Any]:
        if len(self.feature_buffer) < int(self.config["window_size"]):
            return {
                "status": "insufficient_data",
                "message": f"Need {int(self.config['window_size']) - len(self.feature_buffer)} more samples",
            }

        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        scaled = self._scale_sequence(sequence)
        probabilities = self.model.predict(
            scaled.reshape(1, int(self.config["window_size"]), len(self.feature_names)),
            verbose=0,
        )[0]

        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        if confidence >= self.confidence_thresholds["high_confidence"]:
            confidence_level = "high"
        elif confidence >= self.confidence_thresholds["medium_confidence"]:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "status": "prediction_ready",
            "predicted_class": predicted_class,
            "predicted_state": self.class_names[predicted_class],
            "confidence": confidence,
            "confidence_level": confidence_level,
            "class_probabilities": {
                self.class_names[index]: float(probability)
                for index, probability in enumerate(probabilities)
            },
            "risk_level": self._assess_risk(predicted_class, confidence, probabilities),
            "recommendations": self._generate_recommendations(predicted_class, confidence),
        }

    def _scale_sequence(self, sequence: np.ndarray) -> np.ndarray:
        reshaped = sequence.reshape(-1, sequence.shape[-1])
        scaled = self.scaler.transform(reshaped)
        return scaled.reshape(sequence.shape)

    def _series(self, key: str) -> List[float]:
        return [float(entry[key]) for entry in self.raw_history]

    @staticmethod
    def _rate(current: float, previous: float, dt: float) -> float:
        return (current - previous) / dt if dt > 0 else 0.0

    @staticmethod
    def _rolling_mean(values: List[float], window: int) -> float:
        subset = values[-window:]
        return float(np.mean(subset)) if subset else 0.0

    @staticmethod
    def _rolling_std(values: List[float], window: int) -> float:
        subset = values[-window:]
        if len(subset) <= 1:
            return 0.0
        return float(np.std(subset, ddof=1))

    @staticmethod
    def _temp_diff_rate(temp_diff_series: List[float], dt: float) -> float:
        if len(temp_diff_series) <= 1:
            return 0.0
        return (temp_diff_series[-1] - temp_diff_series[-2]) / dt if dt > 0 else 0.0

    def _assess_risk(self, predicted_class: int, confidence: float, probabilities: np.ndarray) -> str:
        if predicted_class == 2:
            if confidence >= 0.8:
                return "critical"
            if confidence >= 0.6:
                return "high"
            return "medium"

        if predicted_class == 1:
            if confidence >= 0.8:
                return "high"
            if confidence >= 0.6:
                return "medium"
            return "low"

        fault_probability = float(probabilities[1] + probabilities[2])
        if fault_probability > 0.3:
            return "medium"
        return "low"

    @staticmethod
    def _generate_recommendations(predicted_class: int, confidence: float) -> List[str]:
        if predicted_class == 2:
            if confidence >= 0.8:
                return [
                    "Potential LOFA detected",
                    "Verify actual coolant flow immediately",
                    "Check pump status and cooling margin",
                ]
            return [
                "Monitor coolant flow closely",
                "Increase fault-monitoring frequency",
            ]

        if predicted_class == 1:
            if confidence >= 0.8:
                return [
                    "Scram-like behavior detected",
                    "Verify shutdown logic and rod response",
                ]
            return [
                "Watch for fast power changes",
                "Verify rod control telemetry",
            ]

        recommendations = ["System operating within normal parameters"]
        if confidence < 0.7:
            recommendations.append("Low confidence; verify sensor quality")
        return recommendations


class FaultDetectionManager(LoggerMixin):
    """Singleton loader for the backend LSTM fault detector."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaultDetectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._detector: Optional[LSTMFaultDetector] = None
        self._load_error: Optional[str] = None

    def _get_detector(self) -> LSTMFaultDetector:
        if self._detector is not None:
            return self._detector

        try:
            self._detector = LSTMFaultDetector(
                model_path=LSTM_MODEL_CONFIG["model_path"],
                config_path=LSTM_MODEL_CONFIG["config_path"],
                scaler_path=LSTM_MODEL_CONFIG["scaler_path"],
            )
            self._load_error = None
            return self._detector
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        detector = self._get_detector()
        return detector.reset(initial_state=initial_state)

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        detector = self._get_detector()
        return detector.update(state)

    def get_last_prediction(self) -> Optional[Dict[str, Any]]:
        if self._detector is None:
            return None
        return self._detector.last_prediction

    def get_error_response(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": self._load_error or "LSTM fault detector unavailable",
        }
