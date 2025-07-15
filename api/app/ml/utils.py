# app/ml/utils.py  (minimal, cross‑platform)

def configure_pytensor_compiler(*_, **__):  # noqa: D401,E501
    """
    Stub kept for backward‑compatibility.

    The project now uses the **JAX backend**, so PyTensor never calls a C
    compiler.  This function therefore does nothing and always returns True.
    """
    return True 
