# Architecture

The repository separates canonical dataset construction, shared preprocessing,
model adapters, training/evaluation, ensemble logic, and the later SegAgent
application. Model adapters may change tensor layout and model-specific padding,
but must not redefine shared preprocessing or ground truth.
