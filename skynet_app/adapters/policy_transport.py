"""Exchange numeric arrays without pickling NumPy implementation classes."""

import io


def encode(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"__skynet_array_npy__": stream.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value):
    import numpy as np

    if isinstance(value, dict):
        if set(value) == {"__skynet_array_npy__"}:
            return np.load(io.BytesIO(value["__skynet_array_npy__"]), allow_pickle=False)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def send_message(connection, value):
    connection.send(encode(value))


def receive_message(connection):
    return decode(connection.recv())
