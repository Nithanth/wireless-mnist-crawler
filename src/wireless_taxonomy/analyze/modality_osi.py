from __future__ import annotations


MODALITY_TO_OSI = {
    "RF measurements": ["L1"],
    "CSI": ["L1"],
    "RSSI": ["L1"],
    "SINR": ["L1"],
    "RSRP": ["L1"],
    "RSRQ": ["L1"],
    "MAC frames": ["L2"],
    "packet traces": ["L3", "L4"],
    "throughput": ["L4", "L7"],
    "QoE metrics": ["L7"],
}


class ModalityOsiMapper:
    provider_name = "modality_osi_rules_v0"

    def map_text(self, text: str) -> tuple[list[str], list[str], float]:
        modalities = [name for name in MODALITY_TO_OSI if name.lower() in text.lower()]
        layers = sorted({layer for name in modalities for layer in MODALITY_TO_OSI[name]})
        return modalities, layers, 0.90 if modalities else 0.50
