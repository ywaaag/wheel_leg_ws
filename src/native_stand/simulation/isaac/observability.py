"""Motion consumers of the public Measurement; no private clock or sink."""

from contextlib import nullcontext


def section(measurement, name, *, parent=None, reason=None):
    if measurement is None or not measurement.enabled:
        return nullcontext()
    if reason is not None:
        measurement.count(name + ".calls", reason=reason)
    return measurement.section(name, parent=parent, synchronize=False)


def count(measurement, name, amount=1, *, reason=None):
    if measurement is not None and measurement.enabled:
        measurement.count(name, amount, reason=reason)


def cpu(value, measurement, *, reason):
    if value.device.type != "cpu":
        count(
            measurement,
            "motion.explicit_transfer_bytes",
            value.numel() * value.element_size(),
            reason=reason,
        )
    return value.cpu()
